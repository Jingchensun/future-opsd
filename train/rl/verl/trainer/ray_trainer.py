# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface.
"""

import json
import math
import os
import re
import subprocess
import uuid
from collections import Counter as collections_counter, defaultdict
from copy import deepcopy
from datetime import datetime
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Optional, Type

import numpy as np
import ray
import torch
from ray.experimental.tqdm_ray import tqdm
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.checkpoint import CHECKPOINT_TRACKER, find_latest_ckpt, remove_obsolete_ckpt
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str, timer, unflatten_dict
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import AutoRewardManager
from .config import PPOConfig
from .core_algos import (
    AdvantageEstimator,
    FixedKLController,
    KLController,
    compute_advantage_return,
    compute_kl,
    get_kl_controller,
)
from .metrics import (
    compute_data_metrics,
    compute_length_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)


class Role(IntEnum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create ray resource pools for distributed training."""
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for different models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker."""
        return self.resource_pool_dict[self.mapping[role]]

    def get_num_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        gpus_available = ray.available_resources().get("GPU", 0)
        gpus_required = self.get_num_gpus()
        if gpus_available < gpus_required:
            raise ValueError(f"Total available GPUs {gpus_available} is less than total desired GPUs {gpus_required}.")


def apply_kl_penalty(data: DataProto, kl_ctrl: KLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards."""
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    # compute kl between ref_policy and current policy
    kld = compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
    kld = kld * response_mask  # (batch_size, response_length)

    data.batch["token_level_rewards"] = token_level_scores - kl_ctrl.kl_coef * kld

    current_kl = torch.mean(VF.masked_mean(kld, mask=response_mask, dim=-1)).item()
    metrics = {"actor/kl_penalty": current_kl, "actor/kl_coef": kl_ctrl.kl_coef}

    # According to https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L880
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return data, metrics


def _ldpo_build_latent_mask(
    input_ids: torch.Tensor,
    response_length: int,
    start_id: int,
    end_id: int,
) -> torch.Tensor:
    """Return a (B, response_length) bool mask True at positions strictly
    inside ``<|latent_start|>...<|latent_end|>`` spans.

    Inlined here (instead of importing from ``workers.actor.dp_actor``)
    so the trainer module stays free of worker-side dependencies.
    """
    bsz, _ = input_ids.shape
    device = input_ids.device
    response_ids = input_ids[:, -response_length:]
    mask = torch.zeros(bsz, response_length, dtype=torch.bool, device=device)

    for b in range(bsz):
        row = response_ids[b]
        starts = (row == start_id).nonzero(as_tuple=False).squeeze(-1)
        ends = (row == end_id).nonzero(as_tuple=False).squeeze(-1)
        if starts.numel() == 0 or ends.numel() == 0:
            continue
        i_ptr, j_ptr = 0, 0
        while i_ptr < starts.numel() and j_ptr < ends.numel():
            s_pos = int(starts[i_ptr].item())
            while j_ptr < ends.numel() and int(ends[j_ptr].item()) <= s_pos:
                j_ptr += 1
            if j_ptr >= ends.numel():
                break
            e_pos = int(ends[j_ptr].item())
            if e_pos > s_pos + 1:
                mask[b, s_pos + 1 : e_pos] = True
            i_ptr += 1
            j_ptr += 1
    return mask


def _attach_reward_metrics_to_batch(data: DataProto, reward_metrics: dict[str, list[float]]) -> None:
    """Persist per-sample reward components for downstream advantage shaping."""
    if data.batch is None or not reward_metrics:
        return
    device = data.batch["responses"].device
    batch_len = len(data)
    for key, values in reward_metrics.items():
        if values is None or len(values) != batch_len:
            continue
        try:
            tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
        except (TypeError, ValueError):
            continue
        safe_key = key.replace("/", "_")
        data.batch[f"reward_{safe_key}_scores"] = tensor


def compute_advantage(data: DataProto, adv_estimator: AdvantageEstimator, gamma: float = 1.0, lam: float = 1.0):
    """Compute advantage estimates for policy optimization."""
    adv_inputs = {
        "token_level_rewards": data.batch["token_level_rewards"],
        "response_mask": data.batch["response_mask"],
        "index": data.non_tensor_batch["uid"],
        "gamma": gamma,
        "lam": lam,
    }
    if "values" in data.batch:
        adv_inputs["values"] = data.batch["values"]

    if "reward_baselines" in data.batch:
        adv_inputs["reward_baselines"] = data.batch["reward_baselines"]

    advantages, returns = compute_advantage_return(adv_estimator, **adv_inputs)
    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        train_dataloader: StatefulDataLoader,
        val_dataloader: StatefulDataLoader,
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[AutoRewardManager] = None,
        val_reward_fn: Optional[AutoRewardManager] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.val_reward_score = 0.0
        self.best_val_reward_score = -1.0
        self.best_global_step = None

        self.hybrid_engine = config.worker.hybrid_engine
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if config.algorithm.disable_kl:
            self.use_reference_policy = False
            self.kl_ctrl = FixedKLController(init_kl_coef=0.0)
            print("KL is disabled, no KL metrics will be logged. Please set `kl_coef=0` to log KL metrics.")
        else:
            self.use_reference_policy = True
            self.kl_ctrl = get_kl_controller(config.algorithm)

        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")

        if config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by actor global batch size.")

        if (
            config.data.rollout_batch_size * config.worker.rollout.n
        ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
            raise ValueError(
                "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
            )

        if self.use_critic:
            if config.data.rollout_batch_size % config.worker.critic.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by critic global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.critic.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by critic micro batch size for experience."
                )

        if (
            config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO)
            and config.worker.rollout.n == 1
        ):
            raise ValueError("GRPO and RLOO algorithm need `config.worker.rollout.n > 1`.")

        if config.trainer.max_steps is not None:
            self.training_steps = config.trainer.max_steps
        elif config.data.mini_rollout_batch_size is not None:
            num_examples = len(train_dataloader) * config.data.mini_rollout_batch_size
            self.training_steps = num_examples // config.data.rollout_batch_size * config.trainer.total_epochs
        else:
            self.training_steps = len(train_dataloader) * config.trainer.total_epochs

        config.worker.actor.optim.training_steps = self.training_steps
        config.worker.critic.optim.training_steps = self.training_steps
        print(f"Total training steps: {self.training_steps}")

        self.save_samples = config.trainer.save_samples
        self.samples_save_dir = config.trainer.samples_save_dir
        self.samples_save_interval = config.trainer.samples_save_interval
        if self.save_samples:
            os.makedirs(self.samples_save_dir, exist_ok=True)
            print(f"Rollout samples will be saved to: {self.samples_save_dir}")

        self._current_rollout_round = 0

    def init_workers(self) -> None:
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor, rollout and ref
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRolloutRef)
            actor_rollout_ref_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRolloutRef], config=self.config.worker, role="actor_rollout_ref"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout_ref"] = actor_rollout_ref_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create a reward model if reward_fn is None
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg: dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_ref_wg = all_wg["actor_rollout_ref"]
        self.actor_rollout_ref_wg.init_model()

    def _save_checkpoint(self) -> None:
        # path: {save_checkpoint_path}/global_step_{global_step}/{actor,critic}
        if self.val_reward_score > self.best_val_reward_score:
            self.best_val_reward_score = self.val_reward_score
            self.best_global_step = self.global_step

        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path,
            self.global_step,
            self.best_global_step,
            self.config.trainer.save_limit,
        )
        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        actor_path = os.path.join(folder_path, "actor")
        self.actor_rollout_ref_wg.save_checkpoint(actor_path, save_model_only=self.config.trainer.save_model_only)

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(critic_path, save_model_only=self.config.trainer.save_model_only)

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)

        checkpointer_tracker_info = {
            "best_global_step": self.best_global_step,
            "best_val_reward_score": round(self.best_val_reward_score, 4),
            "last_global_step": self.global_step,
            "last_actor_path": os.path.abspath(actor_path),
        }
        checkpointer_tracker_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(checkpointer_tracker_path, "w") as f:
            json.dump(checkpointer_tracker_info, f, ensure_ascii=False, indent=2)

    def _load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is not None:
            load_checkpoint_path = self.config.trainer.load_checkpoint_path
        elif self.config.trainer.find_last_checkpoint:
            load_checkpoint_path, tracker_info = find_latest_ckpt(self.config.trainer.save_checkpoint_path)
            if tracker_info is not None:
                self.best_val_reward_score = tracker_info.get("best_val_reward_score", 0.0)
                self.best_global_step = tracker_info.get("best_global_step", 0)
        else:
            load_checkpoint_path = None

        if load_checkpoint_path is None:
            return

        if "global_step_" not in load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Load from checkpoint: {load_checkpoint_path}.")
        self.global_step = int(load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        actor_path = os.path.join(load_checkpoint_path, "actor")
        self.actor_rollout_ref_wg.load_checkpoint(actor_path)
        if self.use_critic:
            critic_path = os.path.join(load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)

        dataloader_path = os.path.join(load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")

    @staticmethod
    def _jsonable(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if torch.is_tensor(value):
            return value.detach().cpu().tolist()
        return value

    def _extract_sample_data(self, batch: DataProto, index: int) -> dict[str, Any]:
        sample_data: dict[str, Any] = {}
        non_tensor_batch = batch.non_tensor_batch if hasattr(batch, "non_tensor_batch") else {}

        uid = non_tensor_batch.get("uid", [""] * len(batch))[index] if "uid" in non_tensor_batch else ""
        sample_data["uid"] = str(uid)

        if "problem" in non_tensor_batch:
            sample_data["prompt"] = self._jsonable(non_tensor_batch["problem"][index])
        elif "raw_prompt_ids" in non_tensor_batch:
            sample_data["prompt"] = self.tokenizer.decode(non_tensor_batch["raw_prompt_ids"][index], skip_special_tokens=False)
        else:
            prompt_ids = None
            if batch.batch is not None:
                if "prompts" in batch.batch:
                    prompt_ids = batch.batch["prompts"][index]
                elif "input_ids" in batch.batch:
                    prompt_ids = batch.batch["input_ids"][index]
            sample_data["prompt"] = (
                self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
                if prompt_ids is not None else ""
            )

        if batch.batch is not None and "responses" in batch.batch:
            response_ids = batch.batch["responses"][index]
            if "response_mask" in batch.batch:
                response_len = int(batch.batch["response_mask"][index].sum().item())
                response_ids = response_ids[:response_len]
            sample_data["response"] = self.tokenizer.decode(response_ids, skip_special_tokens=False)
            sample_data["token_ids"] = response_ids.detach().cpu().tolist()
        else:
            sample_data["response"] = ""
            sample_data["token_ids"] = []

        if "ground_truth" in non_tensor_batch:
            sample_data["ground_truth"] = self._jsonable(non_tensor_batch["ground_truth"][index])
        else:
            sample_data["ground_truth"] = ""

        for key, values in non_tensor_batch.items():
            if key in {"uid", "problem", "raw_prompt_ids", "multi_modal_data", "ground_truth"}:
                continue
            sample_data[key] = self._jsonable(values[index])

        return sample_data

    def _add_sample_scores(self, sample_data: dict[str, Any], batch: DataProto, index: int) -> None:
        if batch.batch is None:
            return
        if "token_level_scores" in batch.batch:
            sample_data["reward_score"] = float(batch.batch["token_level_scores"][index].sum().item())
        if "advantages" in batch.batch:
            advantages = batch.batch["advantages"][index]
            if "response_mask" in batch.batch:
                mask = batch.batch["response_mask"][index].bool()
                advantages = advantages[mask]
            sample_data["avg_advantage"] = float(advantages.mean().item()) if advantages.numel() > 0 else 0.0

    def _group_samples(self, batch: DataProto) -> list[dict[str, Any]]:
        uid_to_samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for i in range(len(batch)):
            sample_data = self._extract_sample_data(batch, i)
            self._add_sample_scores(sample_data, batch, i)
            uid_to_samples[sample_data["uid"]].append(sample_data)

        groups = []
        for uid, samples in uid_to_samples.items():
            group_data: dict[str, Any] = {
                "uid": uid,
                "prompt": samples[0]["prompt"] if samples else "",
                "ground_truth": samples[0]["ground_truth"] if samples else "",
                "samples": samples,
            }
            for source_key, avg_key in (
                ("reward_score", "avg_reward_score"),
                ("avg_advantage", "avg_advantage"),
            ):
                values = [sample[source_key] for sample in samples if source_key in sample]
                if values:
                    group_data[avg_key] = sum(values) / len(values)
            groups.append(group_data)
        return groups

    def _save_samples_json(self, batch: DataProto, step: int, filename: str, extra: Optional[dict[str, Any]] = None) -> None:
        if not self.save_samples or self.samples_save_interval <= 0 or step % self.samples_save_interval != 0:
            return
        try:
            groups = self._group_samples(batch)
            payload = {
                "step": step,
                "timestamp": datetime.now().isoformat(),
                "total_samples": len(batch),
                "total_groups": len(groups),
                "groups": groups,
            }
            if extra:
                payload.update(extra)

            filepath = os.path.join(self.samples_save_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"Saved {len(batch)} rollout samples ({len(groups)} groups) to {filepath}")
        except Exception as exc:  # noqa: BLE001
            print(f"Error saving rollout samples: {exc}")

    def _save_round_samples(self, batch: DataProto, step: int, round_num: int) -> None:
        self._save_samples_json(
            batch,
            step,
            filename=f"step{step}_round_{round_num}.json",
            extra={"round": round_num, "stage": "round"},
        )

    def _save_training_samples(self, batch: DataProto, step: int) -> None:
        self._save_samples_json(
            batch,
            step,
            filename=f"step{step}_training.json",
            extra={"stage": "training"},
        )

    def _maybe_log_val_generations(
        self, inputs: list[str], outputs: list[str], labels: list[str], scores: list[float]
    ) -> None:
        """Log a table of validation samples"""
        if self.config.trainer.val_generations_to_log <= 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, labels, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[: self.config.trainer.val_generations_to_log]
        self.logger.log_generation(samples, self.global_step)

    def _validate(self) -> dict[str, Any]:
        reward_tensor_lst = []
        # Lists to collect samples for the table
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
        reward_metrics_lst = defaultdict(list)
        length_metrics_lst = defaultdict(list)
        print("Start validation...")
        self.actor_rollout_ref_wg.prepare_rollout_engine()
        for batch_dict in self.val_dataloader:
            test_batch = DataProto.from_single_dict(batch_dict)
            test_gen_batch = test_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
            )
            repeat_times = self.config.worker.rollout.val_override_config.get("n", 1)
            test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
            test_gen_batch.meta_info["min_pixels"] = self.config.data.min_pixels
            test_gen_batch.meta_info["max_pixels"] = self.config.data.max_pixels
            test_gen_batch.meta_info["image_min_pixels"] = self.config.data.image_min_pixels
            test_gen_batch.meta_info["image_max_pixels"] = self.config.data.image_max_pixels
            test_gen_batch.meta_info["video_min_pixels"] = self.config.data.video_min_pixels
            test_gen_batch.meta_info["video_max_pixels"] = self.config.data.video_max_pixels
            test_gen_batch.meta_info["video_fps"] = self.config.data.video_fps
            test_gen_batch.meta_info["max_video_frames"] = self.config.data.max_video_frames

            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_ref_wg.world_size)
            test_output_gen_batch = self.actor_rollout_ref_wg.generate_sequences(test_gen_batch)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size * repeat_times)

            # repeat to align with repeated responses in rollout
            test_batch = test_batch.repeat(repeat_times=repeat_times, interleave=True)
            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            reward_tensor, reward_metrics = ray.get(self.val_reward_fn.compute_reward.remote(test_batch))

            # store generations
            input_ids = test_batch.batch["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            output_ids = test_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_inputs.extend(input_texts)
            sample_outputs.extend(output_texts)
            sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            for key, value in reward_metrics.items():
                reward_metrics_lst[key].extend(value)

            for key, value in compute_length_metrics(test_batch).items():
                length_metrics_lst[key].append(value)

        self.actor_rollout_ref_wg.release_rollout_engine()
        self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)
        self.val_reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}
        val_length_metrics = {f"val_{key}": value for key, value in reduce_metrics(length_metrics_lst).items()}
        print("Finish validation.")
        return {"val/reward_score": self.val_reward_score, **val_reward_metrics, **val_length_metrics}

    def _futurebench_step_output_dir(self, ckpt_dir: str) -> str:
        step_name = os.path.basename(ckpt_dir.rstrip(os.sep))
        base = os.environ.get("FUTUREBENCH_LMMS_OUTPUT_BASE", "")
        if base:
            return os.path.join(base, step_name)
        return ckpt_dir

    def _parse_futurebench_metrics(self, ckpt_dir: str) -> dict[str, float]:
        rl_v2_root = os.environ.get(
            "FUTURE_L1_RL_V2_ROOT",
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
        )
        parser = os.path.join(rl_v2_root, "tools", "parse_futurebench_lmms_results.py")
        step_out = self._futurebench_step_output_dir(ckpt_dir)
        if not os.path.isfile(parser):
            print(f"[futurebench] parser not found: {parser}")
            return {}
        try:
            proc = subprocess.run(
                [os.environ.get("PYTHON", "python"), parser, step_out, "--json"],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            print(f"[futurebench] failed to run parser: {exc}")
            return {}
        if proc.returncode != 0:
            print(f"[futurebench] parser failed ({proc.returncode}): {proc.stderr.strip()}")
            return {}
        try:
            metrics = json.loads(proc.stdout)
        except json.JSONDecodeError:
            print("[futurebench] parser returned invalid JSON")
            return {}
        return {str(k): float(v) for k, v in metrics.items()}

    def _maybe_futurebench_eval(self) -> Optional[dict[str, float]]:
        freq = int(getattr(self.config.trainer, "futurebench_eval_freq", -1) or -1)
        if freq <= 0 or self.global_step % freq != 0:
            return None

        script = getattr(self.config.trainer, "futurebench_eval_script", None) or os.environ.get(
            "FUTUREBENCH_EVAL_SCRIPT"
        )
        if not script:
            print("[futurebench] skipped: futurebench_eval_script is not set")
            return None
        script = os.path.abspath(os.path.expanduser(str(script)))
        if not os.path.isfile(script):
            print(f"[futurebench] skipped: script not found: {script}")
            return None

        ckpt_dir = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        if not os.path.isdir(os.path.join(ckpt_dir, "actor")):
            print(f"[futurebench] skipped: checkpoint not ready at {ckpt_dir}")
            return None

        print(f"[futurebench] lmms_eval at global_step={self.global_step} ({script})")
        self.actor_rollout_ref_wg.offload_for_external_eval()
        if self.use_critic:
            self.critic_wg.offload_for_external_eval()

        env = os.environ.copy()
        env["CHECKPOINT_DIR"] = ckpt_dir
        env["FUTUREBENCH_EVAL_STEP"] = str(self.global_step)
        env.setdefault(
            "LMMS_STEP_OUT",
            self._futurebench_step_output_dir(ckpt_dir),
        )
        try:
            proc = subprocess.run(["bash", script], env=env, check=False)
            if proc.returncode != 0:
                print(f"[futurebench] eval script exited with {proc.returncode}")
        finally:
            if self.use_critic:
                self.critic_wg.reload_after_external_eval()
            self.actor_rollout_ref_wg.reload_after_external_eval()
            self.actor_rollout_ref_wg.prepare_rollout_engine()

        metrics = self._parse_futurebench_metrics(ckpt_dir)
        if metrics:
            print(f"[futurebench] metrics: {metrics}")
        return metrics or None

    def _balance_batch(self, batch: DataProto, metrics: dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_ref_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _group_normalized_reward(self, batch, metrics):
        """GRPO advantage per row: (r - group mean) / (group std + 1e-6), r = the summed token-level reward,
        groups = uid (the rollouts of one item). Groups with identical rewards get 0."""
        r = batch.batch["token_level_scores"].sum(-1).float()
        uids = batch.non_tensor_batch["uid"]
        groups = defaultdict(list)
        for i, u in enumerate(uids):
            groups[u].append(i)
        out = torch.zeros_like(r)
        flat = 0
        for idx in groups.values():
            g = r[idx]
            if len(idx) < 2 or float(g.std()) < 1e-6:
                flat += 1
                continue
            out[idx] = (g - g.mean()) / (g.std() + 1e-6)
        metrics["opsd/outcome_reward"] = r.mean().item()
        metrics["opsd/outcome_flat_group_frac"] = flat / max(len(groups), 1)
        return out

    # E20 diagnostic: half of the correct hinted responses cite the hint ("the next frames show ...", "in the subsequent
    # frames ..."), so any mention of future / later frames or images rejects the response
    _HINT_LEAK = re.compile(r"future|privileged|frame\s*\d|\b(next|subsequent|later|following|upcoming|additional|extra|final|last)"
                            r"\s+(\d+\s+|few\s+|three\s+)?(frames?|images?|clips?|observations?)\b", re.IGNORECASE)

    def _hint_resample(self, new_batch: DataProto, gen_batch: DataProto, regular_batch: DataProto,
                       all_metrics: dict, hint_metrics: dict) -> None:
        """E20a (HINT_RESAMPLE=1): future-hinted re-sampling of GRPO's zero-signal groups, in place.

        Groups whose n rollouts are all wrong get HINT_N more samples from the same policy prompted with the true future
        (the OPSD teacher view, dataset OPSD_TEACHER=1; HINT_VIEW=student: from x itself, the more-samples control); up to HINT_MAX_REPLACE verified-correct ones that do not cite the
        hint (_HINT_LEAK) replace wrong rollouts of the group. They are trained as responses to the student prompt x (the
        rollout returns x + response; old_log_probs are then computed under x), so the group gets a GRPO signal.
        The rewards are computed here (token_level_scores), reward/* logs the student's own rollouts only."""
        n = self.config.worker.rollout.n
        reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(regular_batch))
        for k, v in reward_metrics.items():
            all_metrics[k].extend(v)
        acc = torch.as_tensor(reward_metrics["accuracy"], dtype=torch.float32).view(-1, n)  # interleaved: (prompts, n)
        hard = torch.nonzero(acc.sum(-1) == 0).squeeze(-1).tolist()
        if "labeled" in new_batch.non_tensor_batch:  # E20b (label-scarce): an unlabeled row's answer may not be used
            hard = [h for h in hard if bool(new_batch.non_tensor_batch["labeled"][h])]
        hint_metrics["hint/all_wrong_frac"].append(len(hard) / acc.size(0))
        n_replaced, n_rescued, n_leak, hint_acc = 0, 0, 0, []
        if hard:
            hint_n, k_max = int(os.environ.get("HINT_N", str(n))), int(os.environ.get("HINT_MAX_REPLACE", "2"))
            ws = self.actor_rollout_ref_wg.world_size
            rows = [hard[i % len(hard)] for i in range(-(-len(hard) // ws) * ws)]  # pad to the worker count
            hint_gen = gen_batch[rows]
            if os.environ.get("HINT_VIEW", "future") == "future":
                hint_gen.non_tensor_batch["raw_prompt_ids"] = new_batch.non_tensor_batch["teacher_raw_prompt_ids"][rows]
                hint_gen.non_tensor_batch["multi_modal_data"] = new_batch.non_tensor_batch["teacher_multi_modal_data"][rows]
            else:  # HINT_VIEW=student (E20b2c control): the same extra samples from x itself, no future
                assert os.environ["HINT_VIEW"] == "student", os.environ["HINT_VIEW"]
            hint_gen.meta_info = {**gen_batch.meta_info, "n": hint_n}  # index_select shares meta_info
            hint_out = self.actor_rollout_ref_wg.generate_sequences(hint_gen)
            # the response is trained under the student's own inputs
            hint_out.non_tensor_batch["multi_modal_data"] = np.repeat(gen_batch.non_tensor_batch["multi_modal_data"][rows], hint_n)
            hint_batch = new_batch[rows].repeat(repeat_times=hint_n, interleave=True).union(hint_out)
            hint_reward, hint_rm = ray.get(self.reward_fn.compute_reward.remote(hint_batch))
            resp_len = hint_batch.batch["response_mask"].sum(-1)
            for p, h in enumerate(hard):  # the first len(hard) blocks of rows are the distinct hard prompts
                ok = []
                for j in range(p * hint_n, (p + 1) * hint_n):
                    hint_acc.append(hint_rm["accuracy"][j])
                    if hint_rm["accuracy"][j] < 1:
                        continue
                    text = self.tokenizer.decode(hint_batch.batch["responses"][j][: int(resp_len[j])], skip_special_tokens=True)
                    if self._HINT_LEAK.search(text):
                        n_leak += 1
                        continue
                    ok.append(j)
                n_rescued += bool(ok)
                for slot, j in enumerate(ok[:k_max]):
                    dst = h * n + slot
                    for key in hint_out.batch.keys():
                        regular_batch.batch[key][dst] = hint_batch.batch[key][j]
                    reward_tensor[dst] = hint_reward[j]
                    n_replaced += 1
        regular_batch.batch["token_level_scores"] = reward_tensor
        hint_metrics["hint/rescued_frac"].append(n_rescued / max(len(hard), 1))
        hint_metrics["hint/replaced_frac"].append(n_replaced / len(regular_batch))
        hint_metrics["hint/leak_rejected"].append(float(n_leak))
        if hint_acc:
            hint_metrics["hint/hinted_acc"].append(float(np.mean(hint_acc)))

    def _opsd_group_weights(self, batch: DataProto, adv: torch.Tensor, mask: torch.Tensor, mode: str,
                            metrics: dict[str, Any]) -> torch.Tensor:
        """E15: per-rollout weights within each item's group (uid for rollout.n > 1; opsd_group for hypothesis codes).

        s_i = mean over response tokens of the clipped log pT - log pS: how consistent rollout i is with the true future
        (per token, a length-normalised version of the importance weight q(z|x,y)/p(z|x) = pT(z|x,y)/pS(z|x)).
          reweight (E15a): w_i = G * softmax_group(standardised s_i / OPSD_SEQ_TAU)   (mean 1 within each group)
          mcl (E15b):      w_i = 1 for the group's best rollout, else 0              (winner-take-all over codes)
        Also logs whether s tracks answer quality: token F1 of the <answer> against the reference answer for the
        group's top / bottom rollout vs the group mean (the E15a go / no-go check).
        """
        assert mode in ("reweight", "mcl"), f"OPSD_GROUP={mode}"
        s = (adv * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
        keys = batch.batch["opsd_group"].tolist() if "opsd_group" in batch.batch else list(batch.non_tensor_batch["uid"])
        groups = defaultdict(list)
        for i, k in enumerate(keys):
            groups[k].append(i)
        texts = self.tokenizer.batch_decode(
            [r[m.bool()] for r, m in zip(batch.batch["responses"], batch.batch["response_mask"])], skip_special_tokens=True
        )

        def answer(t):
            a = t.split("<answer>")[-1].split("</answer>")[0] if "<answer>" in t else ""
            return " ".join("".join(c if c.isalnum() else " " for c in a.lower()).split())

        def f1(a, b):
            a, b = a.split(), b.split()
            common = sum((collections_counter(a) & collections_counter(b)).values())
            return 0.0 if common == 0 else 2 * common / (len(a) + len(b))

        gts = batch.non_tensor_batch.get("ground_truth")
        ans = [answer(t) for t in texts]
        f = [f1(a, answer("<answer>" + str(g) + "</answer>")) for a, g in zip(ans, gts)] if gts is not None else None
        tau = float(os.environ.get("OPSD_SEQ_TAU", "1.0"))
        w = torch.zeros_like(s)
        stat = defaultdict(list)
        for idx in groups.values():
            it = torch.tensor(idx)
            sg = s[it]
            stat["s_spread"].append(sg.std(unbiased=False).item())
            if mode == "reweight":
                p = torch.softmax((sg - sg.mean()) / (sg.std(unbiased=False) + 1e-6) / tau, dim=0)
                w[it] = p * len(idx)
                stat["ess_frac"].append((1.0 / (p * p).sum()).item() / len(idx))
                stat["w_max"].append(p.max().item() * len(idx))
            else:
                w[it[sg.argmax()]] = 1.0
                stat["distinct_answers"].append(len({ans[i] for i in idx}) / len(idx))
            if f is not None and len(idx) > 1:
                fg = [f[i] for i in idx]
                stat["f1_top_minus_mean"].append(fg[int(sg.argmax())] - sum(fg) / len(fg))
                stat["f1_bottom_minus_mean"].append(fg[int(sg.argmin())] - sum(fg) / len(fg))
                stat["f1_mean"].append(sum(fg) / len(fg))
        for k, v in stat.items():
            metrics[f"opsd_group/{k}"] = float(np.mean(v))
        metrics["opsd_group/n_groups"] = float(len(groups))
        if mode == "mcl" and "opsd_code" in batch.batch:
            codes = batch.batch["opsd_code"]
            for c in sorted(set(codes.tolist())):
                metrics[f"opsd_group/code{c}_win"] = (w[codes == c] > 0).float().mean().item()
        return w

    def _compute_opsd_teacher_log_probs(self, batch: DataProto, prefix: str = "teacher") -> torch.Tensor:
        """Log-probs of the student's sampled response under the teacher prompt (student prompt + future frames).

        The teacher is the current policy itself (same weights, no gradient). Its sequence is
        [left-padded teacher prompt | student response]; response position ids continue from the
        teacher prompt, exactly as the rollout does for the student prompt.
        """
        responses = batch.batch["responses"]
        resp_mask = batch.batch["response_mask"]
        t_ids = batch.batch[f"{prefix}_input_ids"]
        t_mask = batch.batch[f"{prefix}_attention_mask"]
        t_pos = batch.batch[f"{prefix}_position_ids"]
        bsz, resp_len = responses.shape
        delta = torch.arange(1, resp_len + 1, device=t_pos.device).view(1, -1).expand(bsz, -1)
        if t_pos.dim() == 3:  # qwen-vl mrope: (bsz, 4, seq)
            delta = delta.view(bsz, 1, -1).expand(bsz, t_pos.size(1), -1)
        teacher = DataProto.from_dict(
            tensors={
                "input_ids": torch.cat([t_ids, responses], dim=-1),
                "attention_mask": torch.cat([t_mask, resp_mask.to(t_mask.dtype)], dim=-1),
                "position_ids": torch.cat([t_pos, t_pos[..., -1:] + delta], dim=-1),
                "responses": responses,
                "response_mask": resp_mask,
            },
            non_tensors={
                "multi_modal_data": batch.non_tensor_batch[f"{prefix}_multi_modal_data"],
                # distinct uids: the worker caches processed multi-modal inputs by uid
                "uid": np.array([f"{u}_{prefix}" for u in batch.non_tensor_batch["uid"]], dtype=object),
            },
            meta_info=dict(batch.meta_info),
        )
        # OPSD_TEACHER_WEIGHTS=ref (default): frozen initial model + future frames -- a fixed anchor.
        # =current: the policy's own current weights (pure self-distillation; collapsed after ~9 steps in our run,
        # because a degrading student drags the shared-weight teacher down with it).
        # OPSD_TEACHER_WEIGHTS=ema: the reference worker holds an EMA of the student (update_ref_ema after each step).
        if os.environ.get("OPSD_TEACHER_WEIGHTS", "ref") == "current":
            return self.actor_rollout_ref_wg.compute_log_probs(teacher).batch["old_log_probs"]
        assert self.use_reference_policy, "OPSD_TEACHER_WEIGHTS=ref needs the reference model (algorithm.disable_kl=false)"
        return self.actor_rollout_ref_wg.compute_ref_log_probs(teacher).batch["ref_log_probs"]

    def _make_batch_data(self, metrics: dict[str, Any]) -> DataProto:
        batch = None
        all_metrics = defaultdict(list)
        filter_metrics = defaultdict(list)
        num_try_make_batch = 0
        self._current_rollout_round = 0
        print("Start generating batch...")
        while True:
            num_try_make_batch += 1
            self._current_rollout_round += 1
            current_round = self._current_rollout_round
            try:
                batch_dict = next(self.data_iterator)
            except StopIteration:
                self.data_iterator = iter(self.train_dataloader)
                batch_dict = next(self.data_iterator)

            meta_info = {
                "min_pixels": self.config.data.min_pixels,
                "max_pixels": self.config.data.max_pixels,
                "image_min_pixels": self.config.data.image_min_pixels,
                "image_max_pixels": self.config.data.image_max_pixels,
                "video_min_pixels": self.config.data.video_min_pixels,
                "video_max_pixels": self.config.data.video_max_pixels,
                "video_fps": self.config.data.video_fps,
                "max_video_frames": self.config.data.max_video_frames,
            }
            new_batch: DataProto = DataProto.from_single_dict(batch_dict, meta_info=meta_info)
            new_batch.non_tensor_batch["uid"] = np.array(
                [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
            )

            # pop those keys for generation
            gen_batch = new_batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                meta_info_keys=[
                    "min_pixels",
                    "max_pixels",
                    "image_min_pixels",
                    "image_max_pixels",
                    "video_min_pixels",
                    "video_max_pixels",
                    "video_fps",
                    "max_video_frames",
                ],
            )

            # generate the trainable rollout batch
            gen_batch_output = self.actor_rollout_ref_wg.generate_sequences(gen_batch)

            if self.config.algorithm.adv_estimator == "remax":
                gen_baseline_batch = deepcopy(gen_batch)
                gen_baseline_batch.meta_info["temperature"] = 0
                gen_baseline_batch.meta_info["n"] = 1
                gen_baseline_output = self.actor_rollout_ref_wg.generate_sequences(gen_baseline_batch)

                new_batch = new_batch.union(gen_baseline_output)
                reward_baseline_tensor, _ = ray.get(self.reward_fn.compute_reward.remote(new_batch))
                reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                new_batch.batch["reward_baselines"] = reward_baseline_tensor
                del gen_baseline_batch, gen_baseline_output

            # repeat to align with repeated responses in rollout
            regular_batch = new_batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
            regular_batch = regular_batch.union(gen_batch_output)
            if os.environ.get("HINT_RESAMPLE", "0") == "1":
                self._hint_resample(new_batch, gen_batch, regular_batch, all_metrics, filter_metrics)

            need_early_reward = (
                self.config.algorithm.online_filtering
                or self.config.algorithm.answer_tag_filtering
            )

            # filter group
            if need_early_reward:
                reward_tensor, reward_metrics = ray.get(self.reward_fn.compute_reward.remote(regular_batch))
                regular_batch.batch["token_level_scores"] = reward_tensor
                _attach_reward_metrics_to_batch(regular_batch, reward_metrics)

                for k, v in reward_metrics.items():
                    all_metrics[k].extend(v)

                self._save_round_samples(regular_batch, self.global_step, current_round)

                uids = regular_batch.non_tensor_batch["uid"]

                # HyLar-style answer-tag filtering: drop a *group* if any of its
                # responses misses or duplicates the <answer>...</answer> tag.
                bad_uids: set = set()
                if self.config.algorithm.answer_tag_filtering:
                    response_ids = regular_batch.batch["responses"]
                    response_length = regular_batch.batch["response_mask"].sum(dim=-1)
                    for i in range(len(regular_batch)):
                        cur_len = int(response_length[i].item())
                        decoded = self.tokenizer.decode(response_ids[i][:cur_len], skip_special_tokens=False)
                        n_tags = decoded.count("</answer>")
                        if n_tags != 1:
                            bad_uids.add(uids[i])

                if self.config.algorithm.online_filtering:
                    filter_scores = reward_metrics[self.config.algorithm.filter_key]
                    uid2scores = defaultdict(list)
                    for uid, score in zip(uids, filter_scores):
                        uid2scores[uid].append(score)

                    uid2mean = {uid: np.mean(scores) for uid, scores in uid2scores.items()}
                    low_uids = [
                        uid for uid, avg_score in uid2mean.items() if avg_score <= self.config.algorithm.filter_low
                    ]
                    high_uids = [
                        uid for uid, avg_score in uid2mean.items() if avg_score >= self.config.algorithm.filter_high
                    ]
                    kept_uids = [
                        uid
                        for uid, avg_score in uid2mean.items()
                        if (
                            avg_score > self.config.algorithm.filter_low
                            and avg_score < self.config.algorithm.filter_high
                            and uid not in bad_uids
                        )
                    ]
                else:
                    # answer_tag_filtering only -> drop the bad groups, keep the rest.
                    uid2mean = {}
                    low_uids = []
                    high_uids = []
                    kept_uids = [uid for uid in set(uids) if uid not in bad_uids]

                total_groups = len(set(uids))
                filter_metrics["filter/total_groups"].append(float(total_groups))
                filter_metrics["filter/kept_groups"].append(float(len(kept_uids)))
                filter_metrics["filter/bad_answer_tag_groups"].append(float(len(bad_uids)))
                filter_metrics["filter/low_score_groups"].append(float(len(low_uids)))
                filter_metrics["filter/high_score_groups"].append(float(len(high_uids)))
                print(
                    "filter_stats: "
                    f"groups={total_groups} kept={len(kept_uids)} "
                    f"bad_answer_tag={len(bad_uids)} low_score={len(low_uids)} high_score={len(high_uids)}",
                    flush=True,
                )
                kept_sample_idxs = [idx for idx, uid in enumerate(uids) if uid in kept_uids]
                if len(kept_sample_idxs) == 0:
                    max_try_make_batch = self.config.trainer.max_try_make_batch
                    if max_try_make_batch > 0 and num_try_make_batch >= max_try_make_batch:
                        print(
                            "No sample is kept after filtering; falling back to the unfiltered "
                            "round because max_try_make_batch has been reached.",
                            flush=True,
                        )
                        filter_metrics["filter/fallback_unfiltered_rounds"].append(1.0)
                        new_batch = regular_batch
                    else:
                        print("No sample is kept after filtering; continue generating.", flush=True)
                        filter_metrics["filter/empty_rounds"].append(1.0)
                        new_batch = None
                else:
                    filter_metrics["filter/fallback_unfiltered_rounds"].append(0.0)
                    filter_metrics["filter/empty_rounds"].append(0.0)
                    new_batch = regular_batch[kept_sample_idxs]
            else:
                new_batch = regular_batch
                self._save_round_samples(new_batch, self.global_step, current_round)

            if new_batch is not None:
                batch = DataProto.concat([batch, new_batch]) if batch is not None else new_batch
            current_batch_size = 0 if batch is None else len(batch) // self.config.worker.rollout.n
            rollout_batch_size = self.config.data.rollout_batch_size
            if current_batch_size < rollout_batch_size:
                print(f"{current_batch_size=} < {rollout_batch_size=}")
                max_try_make_batch = self.config.trainer.max_try_make_batch
                if max_try_make_batch <= 0 or num_try_make_batch < max_try_make_batch:
                    print(f"{num_try_make_batch=}. Continue generating...")
                else:
                    raise RuntimeError(
                        f"{num_try_make_batch=} >= {max_try_make_batch=}. Generated too many. Please check your data."
                    )
            else:
                print(f"{current_batch_size=} >= {rollout_batch_size=}. Finish generating.")
                # Log reward-function breakdown (format / accuracy / …) whenever the
                # filter path computed rewards early. Previously this only ran under
                # `online_filtering`, so GRPO runs with answer-tag filtering alone
                # never emitted `reward/*` scalars despite `token_level_scores` being set.
                if all_metrics:
                    metrics.update({f"reward/{k}": v for k, v in reduce_metrics(all_metrics).items()})
                if filter_metrics:
                    metrics.update(reduce_metrics(filter_metrics))

                final_batch = batch[: self.config.data.rollout_batch_size * self.config.worker.rollout.n]
                self._save_training_samples(final_batch, self.global_step)
                return final_batch

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        self.global_step = 0
        main_tqdm = tqdm(range(self.training_steps), desc="Running step", position=0)
        val_metrics: Optional[dict[str, Any]] = None

        # load checkpoint before doing anything
        self._load_checkpoint()
        main_tqdm.update(self.global_step)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        self.data_iterator = iter(self.train_dataloader)
        while self.global_step < self.training_steps:
            self.global_step += 1

            metrics, timing_raw = {}, {}
            with timer("step", timing_raw):
                # make a batch of data
                with timer("gen", timing_raw):
                    self.actor_rollout_ref_wg.prepare_rollout_engine()
                    batch = self._make_batch_data(metrics=metrics)
                    self.actor_rollout_ref_wg.release_rollout_engine()

                # balance the number of valid tokens on each dp rank.
                # NOTE: this breaks the order of data inside the batch.
                # Please take care when you implement group based adv computation such as GRPO and rloo
                self._balance_batch(batch, metrics=metrics)

                # compute global valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                # compute reward
                if "token_level_scores" not in batch.batch:
                    with timer("reward", timing_raw):
                        reward_ref = self.reward_fn.compute_reward.remote(batch)

                # recompute old_log_probs
                with timer("old", timing_raw):
                    old_log_probs = self.actor_rollout_ref_wg.compute_log_probs(batch)
                    batch = batch.union(old_log_probs)

                # OPSD: the same policy, conditioned on privileged future frames, scores the student's tokens
                if os.environ.get("OPSD_MODE", "0") == "1":
                    with timer("teacher", timing_raw):
                        n_views = len([v for v in os.environ.get("OPSD_VIEWS", "").split(",") if v.strip()])
                        if float(os.environ.get("OPSD_DISTILL_WEIGHT", "1")) == 0:
                            # E20: distillation weight 0 (outcome-only control): every view = the student, advantage 0
                            for v in range(max(n_views, 1)):
                                batch.batch["teacher_log_probs" if v == 0 else f"teacher{v}_log_probs"] = (
                                    batch.batch["old_log_probs"].clone())
                        else:
                            batch.batch["teacher_log_probs"] = self._compute_opsd_teacher_log_probs(batch)
                            # OPSD_VIEWS with more than one view: score the same response under every teacher view
                            for v in range(1, n_views):
                                batch.batch[f"teacher{v}_log_probs"] = self._compute_opsd_teacher_log_probs(
                                    batch, prefix=f"teacher{v}"
                                )

                # compute ref_log_probs
                if self.use_reference_policy:
                    with timer("ref", timing_raw):
                        if os.environ.get("OPSD_MODE", "0") == "1" and float(self.config.algorithm.kl_coef) == 0:
                            # OPSD keeps the reference worker only as the teacher; with kl_coef 0 the KL term is
                            # multiplied by 0, so skip its forward (~41 s/step in E19)
                            batch.batch["ref_log_probs"] = batch.batch["old_log_probs"].clone()
                        else:
                            ref_log_probs = self.actor_rollout_ref_wg.compute_ref_log_probs(batch)
                            batch = batch.union(ref_log_probs)

                # compute values
                if self.use_critic:
                    with timer("values", timing_raw):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)

                with timer("adv", timing_raw):
                    if "token_level_scores" not in batch.batch:
                        # get token level scores asynchronously
                        reward_tensor, reward_metrics = ray.get(reward_ref)
                        batch.batch["token_level_scores"] = reward_tensor
                        _attach_reward_metrics_to_batch(batch, reward_metrics)
                        reward_metrics = {f"reward/{k}": v for k, v in reduce_metrics(reward_metrics).items()}
                        metrics.update(reward_metrics)

                    # apply kl penalty if available
                    if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                        # apply kl penalty to reward
                        batch, kl_metrics = apply_kl_penalty(batch, self.kl_ctrl, self.config.algorithm.kl_penalty)
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    # compute advantages, executed on the driver process
                    if os.environ.get("OPSD_MODE", "0") == "1":
                        # on-policy self-distillation: per-token advantage = log pi_teacher - log pi_student
                        # on the student's own sampled tokens (sampled estimator of -reverse KL)
                        mask = batch.batch["response_mask"].float()
                        n_views = len([v for v in os.environ.get("OPSD_VIEWS", "").split(",") if v.strip()])
                        view_lps = [batch.batch["teacher_log_probs"]] + [
                            batch.batch[f"teacher{v}_log_probs"] for v in range(1, n_views)
                        ]
                        combine = os.environ.get("OPSD_COMBINE", "single")
                        if combine in ("mix", "wmix", "wmix_contrast"):
                            # teacher = the per-token arithmetic mean of the views' distributions (sampled token),
                            # over each sample's distinct views only (duplicates get weight 0);
                            # wmix: a weighted mean, per-sample view weights set by the dataset ({prefix}_weight)
                            prefixes = ["teacher"] + [f"teacher{v}" for v in range(1, n_views)]
                            dup = torch.stack([batch.batch[f"{p}_dup"].bool() for p in prefixes])  # (V, bsz)
                            n_distinct = (~dup).sum(0).clamp(min=1).to(view_lps[0].dtype)  # (bsz,)
                            if combine in ("wmix", "wmix_contrast"):
                                # (wmix_contrast: the last view is a placebo with weight 0, outside the mixture)
                                w = torch.stack([batch.batch[f"{p}_weight"].to(view_lps[0].dtype) for p in prefixes])
                                w = w.masked_fill(dup, 0.0)
                                w = w / w.sum(0, keepdim=True).clamp(min=1e-8)  # (V, bsz)
                                teacher_lp = torch.logsumexp(torch.stack(view_lps) + w.log().unsqueeze(-1), dim=0)
                                for v in range(n_views):
                                    metrics[f"opsd/view{v}_weight"] = w[v].mean().item()
                            else:
                                stacked = torch.stack(view_lps).masked_fill(dup.unsqueeze(-1), float("-inf"))
                                teacher_lp = torch.logsumexp(stacked, dim=0) - n_distinct.log().unsqueeze(-1)
                            batch.batch["teacher_log_probs"] = teacher_lp
                            metrics["opsd/n_distinct_views"] = n_distinct.mean().item()
                            for v, lp in enumerate(view_lps):
                                metrics[f"opsd/view{v}_reverse_kl"] = (
                                    -((lp - batch.batch["old_log_probs"]) * mask).sum() / mask.sum().clamp(min=1.0)
                                ).item()
                        log_ratio = batch.batch["teacher_log_probs"] - batch.batch["old_log_probs"]
                        clip = float(os.environ.get("OPSD_ADV_CLIP", "5.0"))
                        if combine in ("contrast", "wmix_contrast"):
                            # placebo-controlled advantage: log pT(. | true privilege) - log pT(. | placebo, view 1);
                            # OPSD_CONTRAST_LAMBDA < 1 mixes the plain advantage back in.
                            # wmix_contrast (E16): the true privilege is the weighted mixture of views 0..V-2
                            # (true future text + Gemini futures), the placebo the last view
                            lam = float(os.environ.get("OPSD_CONTRAST_LAMBDA", "1.0"))
                            if combine == "wmix_contrast":
                                contrast = batch.batch["teacher_log_probs"] - view_lps[-1]
                            else:
                                contrast = view_lps[0] - view_lps[1]
                            adv = (lam * contrast + (1.0 - lam) * log_ratio).clamp(-clip, clip)
                            n_c = mask.sum().clamp(min=1.0)
                            metrics["opsd/contrast"] = ((contrast * mask).sum() / n_c).item()
                            metrics["opsd/contrast_abs"] = ((contrast.abs() * mask).sum() / n_c).item()
                            metrics["opsd/placebo_reverse_kl"] = (-((view_lps[-1 if combine == "wmix_contrast" else 1] - batch.batch["old_log_probs"]) * mask).sum() / n_c).item()
                        else:
                            assert combine in ("single", "mix", "wmix"), f"OPSD_COMBINE={combine}"
                            adv = log_ratio.clamp(-clip, clip)
                        # E15 (OPSD_GROUP=reweight|mcl): per-rollout weights within each item's group of rollouts
                        group_mode = os.environ.get("OPSD_GROUP", "")
                        row_w = self._opsd_group_weights(batch, adv, mask, group_mode, metrics) if group_mode else None
                        # E20b (OPSD_LABEL_MASK=1): rows with a label (dataset field "labeled") are trained by the outcome
                        # advantage only, rows without one by the distillation advantage only
                        lab, dmask = None, mask
                        if os.environ.get("OPSD_LABEL_MASK", "0") == "1":
                            lab = torch.as_tensor(batch.non_tensor_batch["labeled"].astype(bool), device=mask.device)
                            dmask = mask * (~lab).to(mask.dtype).unsqueeze(-1)
                            metrics["opsd/labeled_frac"] = lab.float().mean().item()
                        if os.environ.get("OPSD_ADV_BASELINE", "batch_mean") == "batch_mean":
                            # E[log pT - log pS] = -KL < 0 pushes every sampled token down (entropy blow-up);
                            # a constant baseline keeps the expected gradient and removes that drift
                            wm = dmask if row_w is None else dmask * row_w.unsqueeze(-1)
                            adv = adv - (adv * wm).sum() / wm.sum().clamp(min=1.0)
                        adv = adv * dmask
                        if row_w is not None:
                            adv = adv * row_w.unsqueeze(-1)
                        # E19: A = outcome advantage + OPSD_DISTILL_WEIGHT * (distillation advantage); weight 0 = plain GRPO
                        w_distill = float(os.environ.get("OPSD_DISTILL_WEIGHT", "1"))
                        metrics["opsd/distill_adv_abs_raw"] = (adv.abs().sum() / mask.sum().clamp(min=1.0)).item()
                        adv = adv * w_distill
                        lam_out = float(os.environ.get("OPSD_OUTCOME_LAMBDA", "0"))
                        if lam_out:
                            # E17: + lambda * the GRPO advantage of the outcome reward (judge), normalised within
                            # each item's group of rollouts (uid), broadcast over the response tokens
                            out_adv = self._group_normalized_reward(batch, metrics)
                            if lab is not None:
                                out_adv = out_adv * lab.to(out_adv.dtype)
                            metrics["opsd/outcome_adv_abs"] = ((lam_out * out_adv.abs().unsqueeze(-1) * mask).sum() / mask.sum().clamp(min=1.0)).item()
                            metrics["opsd/distill_adv_abs"] = (adv.abs().sum() / mask.sum().clamp(min=1.0)).item()
                            adv = adv + lam_out * out_adv.unsqueeze(-1) * mask
                        batch.batch["advantages"] = adv
                        batch.batch["returns"] = adv
                        n_tok = mask.sum().clamp(min=1.0)
                        metrics["opsd/reverse_kl"] = (-(log_ratio * mask).sum() / n_tok).item()
                        metrics["opsd/teacher_logp"] = ((batch.batch["teacher_log_probs"] * mask).sum() / n_tok).item()
                        metrics["opsd/student_logp"] = ((batch.batch["old_log_probs"] * mask).sum() / n_tok).item()
                        metrics["opsd/adv_abs"] = ((adv.abs()).sum() / n_tok).item()
                        metrics["opsd/adv_clipped_frac"] = (((log_ratio.abs() > clip).float() * mask).sum() / n_tok).item()
                        if group_mode == "mcl":
                            # winner-take-all: train on each item's winning hypothesis code only (as many rows as
                            # items, = a plain OPSD step's batch)
                            keep = torch.nonzero(row_w > 0).squeeze(-1).tolist()
                            batch = batch[keep]
                            self._balance_batch(batch, metrics=metrics, logging_prefix="mcl_seqlen")
                            batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    else:
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                        )

                # update critic
                if self.use_critic:
                    with timer("update_critic", timing_raw):
                        critic_output = self.critic_wg.update_critic(batch)

                    critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                    metrics.update(critic_metrics)

                # update actor
                if self.config.trainer.critic_warmup <= self.global_step:
                    with timer("update_actor", timing_raw):
                        actor_output = self.actor_rollout_ref_wg.update_actor(batch)

                    actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                    metrics.update(actor_metrics)

                    # OPSD with an EMA teacher: the reference (teacher) weights track the student
                    if os.environ.get("OPSD_MODE", "0") == "1" and os.environ.get("OPSD_TEACHER_WEIGHTS") == "ema":
                        with timer("ema", timing_raw):
                            self.actor_rollout_ref_wg.update_ref_ema(float(os.environ.get("OPSD_EMA_DECAY", "0.98")))

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.val_freq > 0
                    and self.global_step % self.config.trainer.val_freq == 0
                ):
                    with timer("validation", timing_raw):
                        val_metrics = self._validate()

                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                    with timer("save_checkpoint", timing_raw):
                        self._save_checkpoint()
                    futurebench_metrics = self._maybe_futurebench_eval()
                    if futurebench_metrics:
                        metrics.update(futurebench_metrics)

            # collect metrics
            num_gpus = self.resource_pool_manager.get_num_gpus()
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))

            self.logger.log(data=metrics, step=self.global_step)
            main_tqdm.update()

        # perform validation after training -- only when periodic validation
        # was on (val_freq > 0) or an initial validation already ran. With
        # `val_freq <= 0` AND `val_before_train: false` this is fully a no-op,
        # which matches the "skip all val" behaviour configured via the YAML.
        if (
            self.val_reward_fn is not None
            and self.config.trainer.val_freq > 0
        ):
            if val_metrics is None or self.global_step % self.config.trainer.val_freq != 0:
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)

            if val_metrics is not None:
                print(f"Final validation metrics:\n{convert_dict_to_str(unflatten_dict(val_metrics))}")

        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            self._save_checkpoint()
            futurebench_metrics = self._maybe_futurebench_eval()
            if futurebench_metrics:
                self.logger.log(data=futurebench_metrics, step=self.global_step)
