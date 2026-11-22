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

import json
import math
import os
import sys
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from qwen_vl_utils.vision_process import fetch_video
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
import re
from . import torch_functional as VF


# ----------------------------------------------------------------------
# FutureL1-specific helpers (system prompt + TwiFF frame loader)
# ----------------------------------------------------------------------
# Mirrors `Future-L1/src/constants.SYSTEM_MESSAGE`; used when the SFT package is
# not importable (e.g. FUTURE_L1_CODE_ROOT unset).
_RL_SYSTEM_MESSAGE_FALLBACK = """You are a multimodal reasoning assistant capable of thinking in textual and visual modes.


Use the following tags to switch your thinking mode:

1.  **Textual Mode**: `<reason>Your textual reasoning process</reason>`
    *   For logical analysis, planning, and verbal thought.

2.  **Visual Mode**: `<|latent_start|>Your visual reasoning process<|latent_end|>`
    *   For mental visualization, imagination and simulation.


**Output Rules**:
*   After all thinking is complete, place the final answer inside `<answer>Your Final Answer</answer>`.
"""

_RL_PLAIN_SYSTEM_MESSAGE = "You are a helpful assistant."

_RL_COT_THINKING_SYSTEM_MESSAGE = (
    "You are a multimodal reasoning assistant. "
    "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
    "The reasoning process MUST BE enclosed within <reason> </reason> tags. "
    "After all thinking is complete, place the final answer inside "
    "`<answer>option's letter from the given choices</answer>`."
)

_RL_STRICT_RFIRST_SYSTEM_MESSAGE = """You are a multimodal reasoning assistant.

You must strictly follow this output format:
<reason>Your first textual reasoning step</reason>
<|latent_start|><|latent|><|latent_end|>
<reason>Your next textual reasoning step after latent visual thinking</reason>
<answer>Your final answer</answer>

Rules:
* Always start with one `<reason>...</reason>` block.
* Use at least one `<|latent_start|>...<|latent_end|>` block.
* Every latent block must be followed by a `<reason>...</reason>` block.
* You may repeat latent + reason blocks if needed.
* Finish with exactly one `<answer>...</answer>` block.
* Do not output any text outside these tags.
"""

_RL_SYSTEM_MESSAGE_CACHE: Optional[str] = None


def hyp_code_suffix(code: int) -> str:
    """E15b hypothesis code k (0-based) as a system-prompt suffix; eval/eval_*.sh append the same text via
    OPSD_SYSTEM_SUFFIX."""
    return f"\n\nFuture hypothesis #{code + 1}."


def _truthy_env(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "y"}


def _future_l1_rl_system_prompt() -> str:
    """Return FutureL1's SFT system message.

    Resolution order:
        1. ``FUTURE_L1_CODE_ROOT`` -> ``src.constants.SYSTEM_MESSAGE``
        2. Inferred Future-L1 root (this file lives at ``Future-L1/RL_v2/verl/utils``)
        3. Built-in fallback string (kept in sync with Future-L1/src/constants.py).
    """
    global _RL_SYSTEM_MESSAGE_CACHE
    if os.environ.get("RL_SYSTEM_PROMPT"):  # explicit override (used by OPSD runs)
        return os.environ["RL_SYSTEM_PROMPT"]
    if _truthy_env("RL_PLAIN_QWEN3VL"):
        return _RL_COT_THINKING_SYSTEM_MESSAGE
    if _truthy_env("FUTURE_L1_RL_PLAIN_SYSTEM"):
        return _RL_PLAIN_SYSTEM_MESSAGE
    if _truthy_env("FUTURE_L1_RL_STRICT_SYSTEM"):
        return _RL_STRICT_RFIRST_SYSTEM_MESSAGE

    if _RL_SYSTEM_MESSAGE_CACHE is not None:
        return _RL_SYSTEM_MESSAGE_CACHE

    roots: List[str] = []
    env_root = os.environ.get("FUTURE_L1_CODE_ROOT", "").strip()
    if env_root:
        roots.append(env_root)
    try:
        inferred = str(Path(__file__).resolve().parents[3] / "sft")  # train/sft holds the `src` package
        if inferred not in roots:
            roots.append(inferred)
    except (IndexError, ValueError):
        pass
    for r in roots:
        if not r or not os.path.isdir(r):
            continue
        if r not in sys.path:
            sys.path.insert(0, r)
        try:
            from src.constants import SYSTEM_MESSAGE  # type: ignore[import-not-found]

            _RL_SYSTEM_MESSAGE_CACHE = SYSTEM_MESSAGE
            return _RL_SYSTEM_MESSAGE_CACHE
        except ImportError:
            continue

    _RL_SYSTEM_MESSAGE_CACHE = _RL_SYSTEM_MESSAGE_FALLBACK
    return _RL_SYSTEM_MESSAGE_CACHE


def _get_frame_indices_uni(num_frames: int, vlen: int, start_frames: int = 1, end_frames: int = 1) -> List[int]:
    if vlen <= 0:
        raise ValueError("Video has no frames.")
    if num_frames <= 0:
        return []
    if vlen <= num_frames:
        indices = list(range(vlen))
        if len(indices) < num_frames:
            indices.extend([indices[-1]] * (num_frames - len(indices)))
        return indices
    start = min(start_frames, vlen)
    end = max(vlen - end_frames, start)
    if num_frames == 1:
        return [(start + end - 1) // 2]
    return np.linspace(start, end - 1, num_frames).round().astype(int).tolist()


def read_twiff_frames_by_clip(video_path: str, clip_indices: List[int]) -> List[ImageObject]:
    """Read TwiFF 1-based frame indices from the fixed 8-frame uniform pool.

    Matches ``Future-L1/RL_v2/verl/utils/dataset.py:read_twiff_frames_by_clip``
    so RL_v2 can train on the same TwiFF JSON shards without re-encoding.
    """
    if not clip_indices:
        return []
    try:
        import decord  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "TwiFF RL data requires the `decord` package to decode videos. "
            "Install in your env or rjob image, e.g.: pip install decord"
        ) from exc

    video_reader = decord.VideoReader(video_path, num_threads=1)
    pool = _get_frame_indices_uni(8, len(video_reader), start_frames=1, end_frames=1)
    selected = [pool[int(i) - 1] for i in clip_indices]
    raw = video_reader.get_batch(selected).asnumpy()
    return [Image.fromarray(raw[i]) for i in range(raw.shape[0])]


def _coerce_conversations(value: Any) -> List[dict]:
    """Normalize TwiFF's ``conversations`` field across shard variants."""
    if value is None:
        return []
    if isinstance(value, list):
        return [v for v in value if isinstance(v, dict)]
    if isinstance(value, dict):
        froms = value.get("from") or []
        values = value.get("value") or []
        return [{"from": f, "value": v} for f, v in zip(froms, values)]
    return []


def _normalize_ground_truth_answer(value: Any) -> str:
    """Normalize GT answer text for reward matching.

    If the label is wrapped as ``<answer>...</answer>``, return the inner text
    (e.g. ``<answer>B</answer>`` -> ``B``). Otherwise return stripped text.
    """
    text = str(value if value is not None else "").strip()
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text


_MCQ_ANSWER_INSTRUCTION = "Please provide only the single option letter within the <answer> </answer> tags."


def collate_fn(features: list[dict[str, Any]]) -> dict[str, Any]:
    """Batch features with mixed/non-uniform metadata keys safely.

    Mixed datasets may not share all non-tensor fields (e.g. only some samples
    carry ``problem_id``). We align every key to batch size by filling missing
    non-tensor entries with ``None`` so ``DataProto`` consistency checks pass.
    """
    all_keys = set()
    for feature in features:
        all_keys.update(feature.keys())

    tensors: dict[str, torch.Tensor] = {}
    non_tensors: dict[str, np.ndarray] = {}

    for key in all_keys:
        values = [feature.get(key, None) for feature in features]
        first_non_none = next((v for v in values if v is not None), None)

        if isinstance(first_non_none, torch.Tensor):
            if any(v is None for v in values):
                raise RuntimeError(
                    f"Tensor key {key!r} missing in part of the batch; "
                    "all tensor fields must exist for every sample."
                )
            tensors[key] = torch.stack(values, dim=0)
        else:
            # Keep metadata as a batch-length object vector. Direct
            # np.array(values, dtype=object) may expand same-length lists such
            # as source_options into shape (B, num_options), then later concat
            # fails when another batch has different option counts.
            arr = np.empty(len(values), dtype=object)
            arr[:] = values
            non_tensors[key] = arr

    return {**tensors, **non_tensors}


def process_image(
    image: Union[dict[str, Any], ImageObject, str], min_pixels: Optional[int], max_pixels: Optional[int]
) -> ImageObject:
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    image.load()  # avoid "Too many open files" errors
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if min_pixels is not None and (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def process_video(
    video: str,
    min_pixels: Optional[int],
    max_pixels: Optional[int],
    video_fps: float,
    max_video_frames: Optional[int] = None,
    return_fps: bool = False,
    return_metadata: bool = False,
) -> Any:
    vision_info = {"video": video, "min_pixels": min_pixels, "max_pixels": max_pixels, "fps": video_fps}
    maxf = None
    if max_video_frames is not None and int(max_video_frames) > 0:
        maxf = int(max_video_frames)
        # qwen_vl_utils accepts `fps` + `max_frames` and computes
        # min(max_frames, frames selected by fps, actual frames) internally.
        # Do not pass `nframes` together with `fps`: that is an assertion error.
        vision_info["max_frames"] = maxf
    video_obj = fetch_video(vision_info, return_video_sample_fps=return_fps, return_video_metadata=return_metadata)
    if maxf is None:
        return video_obj

    def _cap_frames(x):
        # Enforce n_frames = min(actual_frames, max_video_frames) post decode.
        # Handles common containers used by qwen_vl_utils across versions.
        try:
            if isinstance(x, list):
                return x[:maxf]
            if isinstance(x, tuple):
                return x[:maxf]
            if isinstance(x, np.ndarray):
                return x[:maxf]
            if torch.is_tensor(x):
                return x[:maxf]
        except Exception:
            return x
        return x

    def _num_frames(x) -> Optional[int]:
        try:
            if isinstance(x, (list, tuple)):
                return len(x)
            if isinstance(x, np.ndarray) or torch.is_tensor(x):
                return int(x.shape[0])
        except Exception:
            return None
        return None

    def _cap_metadata(metadata: dict, n_frames: Optional[int]) -> dict:
        if n_frames is None:
            return metadata
        metadata = dict(metadata)
        indices = metadata.get("frames_indices")
        if isinstance(indices, list):
            metadata["frames_indices"] = indices[:n_frames]
        elif torch.is_tensor(indices):
            metadata["frames_indices"] = indices[:n_frames]
        elif isinstance(indices, np.ndarray):
            metadata["frames_indices"] = indices[:n_frames]
        return metadata

    def _cap_video_obj(obj):
        # return_video_metadata=True returns (video, metadata).
        if isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[1], dict):
            capped_video = _cap_frames(obj[0])
            return capped_video, _cap_metadata(obj[1], _num_frames(capped_video))

        # return_video_sample_fps=True wraps the video object as (video_obj, sample_fps).
        if isinstance(obj, tuple) and len(obj) == 2 and isinstance(obj[1], (float, int)):
            return _cap_video_obj(obj[0]), obj[1]

        return _cap_frames(obj)

    return _cap_video_obj(video_obj)


_LEADING_VIDEO_IMAGE_BLOCK_RE = re.compile(
    r"^\s*<video>\s*(?:\r?\n)\s*<image>\s*(?:\r?\n)?",
    re.IGNORECASE,
)


def _strip_leading_video_image_tags(prompt: str) -> str:
    """Strip leading <video>/<image> lines so HF chat media slots bind without duplicate markers."""
    return _LEADING_VIDEO_IMAGE_BLOCK_RE.sub("", prompt, count=1).lstrip()


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        video_key: str = "videos",
        image_dir: Optional[str] = None,
        video_fps: float = 2.0,
        max_video_frames: Optional[int] = 16,
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        image_min_pixels: Optional[int] = None,
        image_max_pixels: Optional[int] = None,
        video_min_pixels: Optional[int] = None,
        video_max_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
        filter_overlong_prompts_workers: int = 16,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.video_key = video_key
        self.image_dir = image_dir
        self.video_fps = video_fps
        self.max_video_frames = max_video_frames
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.image_min_pixels = image_min_pixels if image_min_pixels is not None else min_pixels
        self.image_max_pixels = image_max_pixels if image_max_pixels is not None else max_pixels
        self.video_min_pixels = video_min_pixels if video_min_pixels is not None else min_pixels
        self.video_max_pixels = video_max_pixels if video_max_pixels is not None else max_pixels

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            # when we use dataset builder, we should always refer to the train split
            file_type = os.path.splitext(os.listdir(data_path)[0])[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_dir=data_path, split=data_split)
        elif os.path.isfile(data_path):
            file_type = os.path.splitext(data_path)[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_files=data_path, split=data_split)
        else:
            # load remote dataset from huggingface hub
            self.dataset = load_dataset(data_path, split=data_split)

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        # Bail out fast if TwiFF JSON was asked for but decord isn't installed.
        self._is_twiff_path = "TwiFF" in data_path or "twiff" in data_path.lower()
        if self._is_twiff_path:
            try:
                import decord  # noqa: F401, PLC0415
            except ImportError as exc:
                raise ImportError(
                    "TwiFF RL data requires the `decord` package to decode videos. "
                    "Install in your env or rjob image, e.g.: pip install decord"
                ) from exc

        if filter_overlong_prompts:
            # TwiFF JSON shards trigger video decode per filter call which is
            # expensive; keep this off by default for TwiFF (matches the
            # upstream future_l1_rl behaviour).
            if self._is_twiff_path and filter_overlong_prompts_workers > 0:
                print(
                    "[RLHFDataset] TwiFF detected; skipping pre-filter to "
                    "avoid full-video decode of every shard.",
                    flush=True,
                )
            else:
                self.dataset = self.dataset.filter(
                    self._filter_overlong_prompts,
                    desc="Filtering overlong prompts",
                    num_proc=filter_overlong_prompts_workers,
                )

    def _encode_videos(self, videos: list) -> tuple[list, dict[str, Any]]:
        """Decode videos with qwen_vl_utils (as the evaluation does) -> (frames per video, extra processor kwargs).

        Qwen3-VL: the HF video processor would resample the decoded frames again (do_sample_frames defaults to True and,
        without metadata, assumes 24 fps -- 32 decoded frames became 4); keep them as decoded and pass the metadata for
        the timestamp tokens, like lmms-eval / vLLM. Other processors: second_per_grid_ts as before."""
        qwen3 = "Qwen3VLProcessor" in self.processor.__class__.__name__
        frames, metas, fps_list = [], [], []
        for video in videos:
            if qwen3:
                v, meta = process_video(video, self.video_min_pixels, self.video_max_pixels, self.video_fps,
                                        self.max_video_frames, return_metadata=True)
                frames.append(v)
                metas.append(meta)
            else:
                v, fps = process_video(video, self.video_min_pixels, self.video_max_pixels, self.video_fps,
                                       self.max_video_frames, return_fps=True)
                frames.append(v)
                fps_list.append(fps)
        if qwen3:
            return frames, {"video_metadata": metas, "do_sample_frames": False}
        if "second_per_grid_ts" in self.processor.model_input_names:
            return frames, {"second_per_grid_ts": [2.0 / f for f in fps_list]}
        return frames, {}

    def _joint_video_image_example(self, example: dict[str, Any]) -> bool:
        """True when both modalities are present (e.g. SEED-Bench-R1 video + current frame)."""
        if self.video_key not in example or self.image_key not in example:
            return False
        v = example.get(self.video_key)
        i = example.get(self.image_key)
        return isinstance(v, list) and isinstance(i, list) and len(v) > 0 and len(i) > 0

    def _build_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        prompt_str: str = example[self.prompt_key]
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            prompt_str = format_prompt.render(content=prompt_str)
        problem_type = str(example.get("problem_type", "")).strip().lower()
        if problem_type == "multiple choice" and _MCQ_ANSWER_INSTRUCTION not in prompt_str:
            prompt_str = f"{prompt_str}\n\n{_MCQ_ANSWER_INSTRUCTION}"

        system_prompt = _future_l1_rl_system_prompt()

        if self._joint_video_image_example(example):
            clean = _strip_leading_video_image_tags(prompt_str)
            videos = example.get(self.video_key) or []
            images = example.get(self.image_key) or []
            content_list: list[dict[str, Any]] = []
            content_list.extend({"type": "video"} for _ in videos)
            content_list.extend({"type": "image"} for _ in images)
            if clean.strip():
                content_list.append({"type": "text", "text": clean})
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_list},
            ]
        if self.image_key in example:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            images = example.get(self.image_key) or []
            num_images = len(images) if isinstance(images, list) else 1
            content_list = []
            if "<image>" not in prompt_str and num_images > 0:
                # Some standard VLM datasets store media paths separately and
                # keep the question text marker-free. Add explicit placeholders
                # so HF/vLLM can bind each media item to prompt tokens.
                content_list.extend({"type": "image"} for _ in range(num_images))
                if prompt_str:
                    content_list.append({"type": "text", "text": prompt_str})
            else:
                for i, content in enumerate(prompt_str.split("<image>")):
                    if i != 0:
                        content_list.append({"type": "image"})

                    if content:
                        content_list.append({"type": "text", "text": content})

            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_list},
            ]
        elif self.video_key in example:
            videos = example.get(self.video_key) or []
            num_videos = len(videos) if isinstance(videos, list) else 1
            content_list = []
            if "<video>" not in prompt_str and num_videos > 0:
                content_list.extend({"type": "video"} for _ in range(num_videos))
                if prompt_str:
                    content_list.append({"type": "text", "text": prompt_str})
            else:
                for i, content in enumerate(prompt_str.split("<video>")):
                    if i != 0:
                        content_list.append({"type": "video"})

                    if content:
                        content_list.append({"type": "text", "text": content})

            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content_list},
            ]
        else:
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_str},
            ]

    def _filter_overlong_prompts(self, example: dict[str, Any]) -> bool:
        # Filter is invoked by HF datasets via num_proc workers; skipping the
        # TwiFF branch above guarantees we never try to spin up decord here.
        if self._is_twiff_example(example):
            return True
        messages = self._build_messages(example)
        if self._joint_video_image_example(example):
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example[self.video_key]
            images = example[self.image_key]
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):
                images = [os.path.join(self.image_dir, image) for image in images]
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_images = []
            for image in images:
                processed_images.append(process_image(image, self.image_min_pixels, self.image_max_pixels))
            processed_videos = []
            video_fps_list = []
            for video in videos:
                processed_video, video_fps = process_video(
                    video,
                    self.video_min_pixels,
                    self.video_max_pixels,
                    self.video_fps,
                    self.max_video_frames,
                    return_fps=True,
                )
                processed_videos.append(processed_video)
                video_fps_list.append(video_fps)

            proc_kw: dict[str, Any] = dict(
                images=processed_images,
                videos=processed_videos,
                text=[prompt],
                add_special_tokens=False,
                return_tensors="pt",
            )
            if "second_per_grid_ts" in self.processor.model_input_names:
                proc_kw["second_per_grid_ts"] = [2.0 / video_sample_fps for video_sample_fps in video_fps_list]

            model_inputs = self.processor(**proc_kw)
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example[self.image_key]
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.image_min_pixels, self.image_max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        elif self.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example[self.video_key]
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            for video in videos:
                processed_videos.append(
                    process_video(
                        video,
                        self.video_min_pixels,
                        self.video_max_pixels,
                        self.video_fps,
                        self.max_video_frames,
                    )
                )

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        else:
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            return len(input_ids) <= self.max_prompt_length

    # ------------------------------------------------------------------
    # FutureL1 / TwiFF-aware row materialization
    # ------------------------------------------------------------------
    def _is_twiff_example(self, example: dict[str, Any]) -> bool:
        return (
            self._is_twiff_path
            or ("conversations" in example and "video" in example and "image" in example)
            or "reasoning_image" in example
        )

    def _materialize_twiff(self, example: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Convert a raw TwiFF row into the standard ``{problem,answer,images}``.

        Returns ``None`` when the row should be skipped (e.g. unreadable video).
        Mirrors ``RL/future_l1_rl/verl/utils/dataset.py:RLHFDataset._build_example``
        for the TwiFF branch.
        """
        video_path = str(example.get("video", "") or "")
        conversations = _coerce_conversations(example.get("conversations"))
        human_turn = None
        gpt_turn = None
        for turn in conversations:
            role = str(turn.get("from", ""))
            if role == "human":
                human_turn = turn
            elif role == "gpt":
                gpt_turn = turn
        if human_turn is None:
            return None

        try:
            image_indices = [int(x) for x in (example.get("image") or [])]
        except (TypeError, ValueError):
            image_indices = []

        # OPSD: the privileged teacher also sees the future frames (reasoning_image), the student does not.
        future_indices: List[int] = []
        if _truthy_env("OPSD_TEACHER"):
            try:
                future_indices = [int(x) for x in (example.get("reasoning_image") or [])]
            except (TypeError, ValueError):
                future_indices = []
            future_indices = future_indices[: int(os.environ.get("OPSD_MAX_FUTURE", "3"))]
        try:
            frames = read_twiff_frames_by_clip(video_path, image_indices + future_indices) if image_indices else []
            images, future_images = frames[: len(image_indices)], frames[len(image_indices) :]
        except ImportError:
            raise
        except Exception as exc:  # noqa: BLE001
            print(f"[RLHFDataset] failed to load TwiFF frames from {video_path}: {exc}", flush=True)
            return None

        problem = str(human_turn.get("value", ""))
        answer = str(example.get("answer", "") or "")
        if not answer and gpt_turn is not None:
            answer = str(gpt_turn.get("value", ""))

        materialized: dict = {}
        if images:
            materialized[self.image_key] = images
            materialized[self.prompt_key] = problem
        else:
            materialized[self.prompt_key] = problem.replace("<image>", "")
        materialized[self.answer_key] = answer
        if _truthy_env("OPSD_TEACHER"):
            materialized["_opsd_future_images"] = future_images
            # text privileged context (OPSD_PRIVILEGED=answer|text): the dataset answer, and the reference
            # reasoning after its first <image> placeholder, i.e. its description of frame_2 onwards
            ref = str(gpt_turn.get("value", "")) if gpt_turn is not None else ""
            materialized["_opsd_answer"] = answer
            materialized["_opsd_future_text"] = ref.split("<image>", 1)[1].replace("<image>", "").strip() if "<image>" in ref else ""
        return materialized

    def _maybe_materialize(self, example: dict[str, Any]) -> Optional[dict[str, Any]]:
        """If the row needs TwiFF-style conversion, do it; else return as-is."""
        if self._is_twiff_example(example):
            return self._materialize_twiff(example)
        if "opsd_future_frames" in example:
            return self._materialize_egoplan(example)
        return example

    def _materialize_egoplan(self, example: dict[str, Any]) -> dict[str, Any]:
        """EgoPlan-IT row (data_tools/make_egoplan_train.py): progress clip + current observation + 4-way question.
        The OPSD privilege is precomputed: future frame files inside the answer action, the next narrated actions
        (text) and the answer ("<letter>. <action>")."""
        example = dict(example)
        paths = example.pop("opsd_future_frames") or []
        future_text, answer_text = example.pop("opsd_future_text", ""), example.pop("opsd_answer_text", "")
        if not example.get(self.video_key):
            example.pop(self.video_key, None)  # no progress yet: an image-only sample
        example["_egoplan"] = True
        if _truthy_env("OPSD_TEACHER"):
            k = int(os.environ.get("OPSD_MAX_FUTURE", "3"))
            example["_opsd_future_images"] = [Image.open(p).convert("RGB") for p in paths[:k]]
            example["_opsd_answer"] = answer_text
            example["_opsd_future_text"] = future_text
        return example

    def getitem_code(self, index: int, code: int):
        """Row `index` with hypothesis code `code` in the student prompt (E15b, HypCodeDataset)."""
        self._hyp_code = code
        try:
            return self[index]
        finally:
            self._hyp_code = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        raw_example: dict = self.dataset[index]
        materialized = self._maybe_materialize(raw_example)
        if materialized is None:
            raise RuntimeError(
                f"TwiFF row at index {index} could not be materialized "
                f"(missing video or unreadable). Filter the shard upstream "
                "or install decord."
            )
        example: dict = materialized
        opsd_future = example.pop("_opsd_future_images", None)
        opsd_text = {"answer": example.pop("_opsd_answer", ""), "text": example.pop("_opsd_future_text", "")}
        if os.environ.get("OPSD_TEXT_CAPTIONS"):  # E30: z=text = a caption of this sample's future frames
            opsd_text["text"] = self._opsd_caption(index)
        if opsd_future is not None and _truthy_env("OPSD_WRONG_FUTURE"):
            # control: the teacher sees the future frames of a different video
            partner = self._maybe_materialize(self.dataset[(index + len(self.dataset) // 2) % len(self.dataset)])
            opsd_future = (partner or {}).get("_opsd_future_images") or []
        opsd_q_images = list(example.get(self.image_key) or [])
        opsd_q_videos = list(example.get(self.video_key) or []) if self._joint_video_image_example(example) else []
        egoplan = bool(example.pop("_egoplan", False))
        messages = self._build_messages(example)
        hyp = getattr(self, "_hyp_code", None)
        if hyp is not None:
            # E15b: the student prompt carries hypothesis code `hyp` as a system-prompt suffix (the same suffix is used at
            # evaluation, OPSD_SYSTEM_SUFFIX); the teacher prompt (_add_opsd_teacher) has no code
            assert messages[0]["role"] == "system", messages[0]["role"]
            messages[0]["content"] += hyp_code_suffix(hyp)
            example["opsd_group"] = torch.tensor(index)
            example["opsd_code"] = torch.tensor(hyp)
        raw_problem_text = example.get(self.prompt_key)
        example.pop(self.prompt_key, None)

        if self._joint_video_image_example(example):
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example.pop(self.image_key)
            videos = example.pop(self.video_key)
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_images = []
            for image in images:
                processed_images.append(process_image(image, self.image_min_pixels, self.image_max_pixels))
            processed_videos, video_kw = self._encode_videos(videos)
            model_inputs = self.processor(images=processed_images, videos=processed_videos, text=[prompt],
                                          add_special_tokens=False, return_tensors="pt", **video_kw)
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"videos": videos, "images": images}
        elif self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example.pop(self.image_key)
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.image_min_pixels, self.image_max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"images": images}
        elif self.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example.pop(self.video_key)
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            video_fps_list = []
            for video in videos:
                processed_video, video_fps = process_video(
                    video,
                    self.video_min_pixels,
                    self.video_max_pixels,
                    self.video_fps,
                    self.max_video_frames,
                    return_fps=True,
                )
                processed_videos.append(processed_video)
                video_fps_list.append(video_fps)

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            if "second_per_grid_ts" in self.processor.model_input_names:
                model_inputs["second_per_grid_ts"] = [2.0 / video_sample_fps for video_sample_fps in video_fps_list]

            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"videos": videos}
        else:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen-vl mrope
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from ..models.transformers.qwen3_vl import get_rope_index
            else:
                from ..models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                attention_mask=attention_mask,
            )  # (3, seq_length)
            text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)  # (1, seq_length)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)  # (4, seq_length)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids
        example["ground_truth"] = _normalize_ground_truth_answer(example.pop(self.answer_key))
        # Keep the raw question text so the LLM-as-judge reward (and any
        # other reward function that wants task context) can access it via
        # `non_tensor_batch["problem"]`.
        if raw_problem_text is not None:
            example["problem"] = raw_problem_text
        if opsd_future is not None:
            views = [v.strip() for v in os.environ.get("OPSD_VIEWS", "").split(",") if v.strip()]
            if not views:
                self._add_opsd_teacher(example, raw_problem_text or "", opsd_q_images, opsd_future, opsd_text,
                                       q_videos=opsd_q_videos, egoplan=egoplan)
            # OPSD_VIEWS: several teacher views of the same sample, stored as teacher_* (view 0), teacher1_*, ...
            #   frames / frames1|2|3 (the first k future frames) / text / answer / frames+answer /
            #   placebo_frames (another video's future frames, as many as this sample's own)
            #   gemink (k = 0, 1, ...): the k-th Gemini-imagined future of this sample (text, OPSD_GEMINI_FUTURES)
            #   placebo_text (another video's reference future text; weight 0 in a wmix, the contrast view of wmix_contrast)
            seen = set()
            raw_w = []  # (is_gemini, weight before normalisation) per view, for OPSD_COMBINE=wmix
            for v, view in enumerate(views):
                text, is_gem = opsd_text, view[:6] == "gemini" and view[6:].isdigit()
                if view == "placebo_frames":
                    mode, future = "frames", self._opsd_placebo_frames(index, len(opsd_future))
                elif view == "placebo_text":
                    mode, future, text = "text", opsd_future, {**opsd_text, "text": self._opsd_placebo_text(index)}
                elif view[:6] == "frames" and view[6:].isdigit():
                    mode, future = "frames", opsd_future[: int(view[6:])]
                elif is_gem:
                    g = self._opsd_gemini_futures(index)[int(view[6:])]
                    mode, future, text = "text", opsd_future, {**opsd_text, "text": g["description"]}
                else:
                    mode, future = view, opsd_future
                prefix = "teacher" if v == 0 else f"teacher{v}"
                self._add_opsd_teacher(example, raw_problem_text or "", opsd_q_images, future, text,
                                       mode=mode, prefix=prefix, q_videos=opsd_q_videos, egoplan=egoplan)
                # a view identical to an earlier one (e.g. frames3 when the sample has a single future frame)
                key = view if is_gem or view == "placebo_text" else (mode, len(future), view == "placebo_frames")
                example[f"{prefix}_dup"] = torch.tensor(key in seen)
                seen.add(key)
                raw_w.append((is_gem, (max(g["probability"], 0.05) if os.environ.get("OPSD_GEMINI_WEIGHTING", "prob") == "prob"
                                       else 1.0) if is_gem else 0.0 if view.startswith("placebo") else 1.0))
            # wmix weights: the true-privilege views share OPSD_TRUE_WEIGHT (default 0.5 when Gemini views are present),
            # the Gemini views share the rest in proportion to Gemini's own probability estimates; placebo views get 0
            n_true, z_gem = sum(not g and w > 0 for g, w in raw_w), sum(w for g, w in raw_w if g)
            w_true = float(os.environ.get("OPSD_TRUE_WEIGHT", "0.5")) if z_gem and n_true else (1.0 if n_true else 0.0)
            for v, (is_gem, w) in enumerate(raw_w):
                prefix = "teacher" if v == 0 else f"teacher{v}"
                example[f"{prefix}_weight"] = torch.tensor((1 - w_true) * w / z_gem if is_gem else w * w_true / max(n_true, 1),
                                                           dtype=torch.float32)
        return example

    def _opsd_gemini_futures(self, index: int) -> list:
        """Gemini-imagined futures of row `index` (an optional multi-future jsonl), [{description, answer, probability}]."""
        if not hasattr(self, "_opsd_gemini"):
            self._opsd_gemini = {}
            for line in open(os.environ["OPSD_GEMINI_FUTURES"]):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._opsd_gemini[int(r["idx"])] = r["futures"]
        if index not in self._opsd_gemini:
            raise RuntimeError(f"no Gemini futures for row {index} in {os.environ['OPSD_GEMINI_FUTURES']}")
        return self._opsd_gemini[index]

    def _opsd_caption(self, index: int) -> str:
        """E30: the Qwen3.5-27B caption of row `index`'s future frames (data_tools/caption_future.py), from the json in
        OPSD_TEXT_CAPTIONS, keyed by str(sample_id) (EgoPlan-IT) or str(id) (Future-L1)."""
        if not hasattr(self, "_opsd_captions"):
            self._opsd_captions = json.load(open(os.environ["OPSD_TEXT_CAPTIONS"]))["captions"]
        row = self.dataset[index]
        key = str(row["sample_id"] if "sample_id" in row else row["id"])
        if key not in self._opsd_captions:
            raise RuntimeError(f"no caption for row {index} ({key}) in {os.environ['OPSD_TEXT_CAPTIONS']}")
        return self._opsd_captions[key]["caption"]

    def _opsd_placebo_text(self, index: int) -> str:
        """The reference future text of a different video (the first partner from index + N/2 on that has one); read
        from the raw conversations (EgoPlan-IT: the raw opsd_future_text column), so no frames are decoded.
        With OPSD_TEXT_CAPTIONS (E30): the partner's caption, same partner rule."""
        n = len(self.dataset)
        own_video = self.dataset[index].get("video_id")
        if os.environ.get("OPSD_TEXT_CAPTIONS"):
            for j in range(64):
                p = (index + n // 2 + j) % n
                if p == index or (own_video is not None and self.dataset[p].get("video_id") == own_video):
                    continue
                text = self._opsd_caption(p)
                if text:
                    return text
            raise RuntimeError(f"no placebo partner with a caption for index {index}")
        for j in range(64):
            p = (index + n // 2 + j) % n
            if p == index:
                continue
            row = self.dataset[p]
            if "opsd_future_text" in row:  # EgoPlan-IT (E21b): the partner's EK-100 narration of the next actions
                if row.get("video_id") != own_video and row["opsd_future_text"]:
                    return row["opsd_future_text"]
                continue
            gpt = [t for t in _coerce_conversations(row.get("conversations")) if str(t.get("from", "")) == "gpt"]
            ref = str(gpt[-1].get("value", "")) if gpt else ""
            if "<image>" in ref:
                text = ref.split("<image>", 1)[1].replace("<image>", "").strip()
                if text:
                    return text
        raise RuntimeError(f"no placebo partner with a future text for index {index}")

    def _opsd_placebo_frames(self, index: int, k: int) -> list:
        """k future frames of a different video (the first partner from index + N/2 on that has >= k)."""
        if k == 0:
            return []
        n = len(self.dataset)
        if not hasattr(self, "_opsd_n_future"):  # future-frame count per row, read from the raw column (no decoding)
            def count(v):
                try:
                    return len(ast.literal_eval(v) if isinstance(v, str) else (v or []))
                except (ValueError, SyntaxError):
                    return 0
            col = "reasoning_image" if "reasoning_image" in self.dataset.column_names else "opsd_future_frames"
            self._opsd_n_future = [count(v) for v in self.dataset[col]]
        tried = 0
        for j in range(n):
            p = (index + n // 2 + j) % n
            if p == index or self._opsd_n_future[p] < k:
                continue
            partner = self._maybe_materialize(self.dataset[p]) or {}
            frames = partner.get("_opsd_future_images") or []
            if len(frames) >= k:
                return frames[:k]
            tried += 1
            if tried >= 8:
                break
        raise RuntimeError(f"no placebo partner with {k} future frames for index {index}")

    def _add_opsd_teacher(self, example: dict, problem: str, q_images: list, future: list, text: dict,
                          mode: Optional[str] = None, prefix: str = "teacher", q_videos: Optional[list] = None,
                          egoplan: bool = False) -> None:
        """Teacher prompt = student prompt + privileged context, encoded exactly like the student's.

        OPSD_PRIVILEGED: frames (default) = the future frames; answer = the dataset answer;
        text = the reference reasoning's description of the future; frames+answer = both.
        q_videos: the student's progress video (EgoPlan-IT rows: video + current observation image); the teacher
        then sees [video, observation image, question + privileged suffix with the future frames inline].
        """
        mode = mode or os.environ.get("OPSD_PRIVILEGED", "frames")
        assert mode in ("frames", "answer", "text", "frames+answer"), f"OPSD_PRIVILEGED={mode}"
        if mode in ("answer", "text"):
            future = []
        k = len(future)
        q_videos = list(q_videos or [])
        head = "\n\n[Privileged context, not visible to the model at test time] "
        made_from = "the video and the current observation" if egoplan else "frame_1"
        tail = "\nUse {} to make your prediction accurate, but write your reasoning as a prediction made from " + made_from + "."
        name = (lambda i: f"future frame {i + 1}") if egoplan else (lambda i: f"frame_{i + 2}")
        if mode == "text":
            suffix = head + "What actually happens later in this video:\n" + text["text"] + tail.format("it")
        elif mode == "answer":
            suffix = head + "The correct answer is: " + text["answer"] + tail.format("it")
        elif k:
            suffix = (
                head + f"The next {k} frame(s) show what actually happens later in this video:\n"
                + "\n".join(f"{name(i)}\n<image>" for i in range(k))
                + ("\nThe correct answer is: " + text["answer"] if mode == "frames+answer" else "")
                + tail.format("them")
            )
        else:
            suffix = ""
        t_images = list(q_images) + list(future)
        t_videos = []
        if q_videos:  # joint video + image student prompt: the suffix's future frames are placed inline after it
            content: list[dict[str, Any]] = [{"type": "video"} for _ in q_videos] + [{"type": "image"} for _ in q_images]
            for i, part in enumerate((_strip_leading_video_image_tags(problem) + suffix).split("<image>")):
                if i:
                    content.append({"type": "image"})
                if part:
                    content.append({"type": "text", "text": part})
            t_messages = [{"role": "system", "content": _future_l1_rl_system_prompt()}, {"role": "user", "content": content}]
            t_videos, t_video_kw = self._encode_videos(q_videos)
        else:
            if egoplan and "<image>" not in problem:  # image-only EgoPlan row: the student gets [image, question]
                problem = "<image>" * len(q_images) + problem
            t_messages = self._build_messages({self.prompt_key: problem + suffix, self.image_key: t_images})
        t_prompt = self.processor.apply_chat_template(t_messages, add_generation_prompt=True, tokenize=False)
        t_processed = [process_image(im, self.image_min_pixels, self.image_max_pixels) for im in t_images] or None
        if t_videos:
            t_inputs = self.processor(images=t_processed, videos=t_videos, text=[t_prompt], add_special_tokens=False,
                                      return_tensors="pt", **t_video_kw)
        else:
            t_inputs = self.processor(t_processed, [t_prompt], add_special_tokens=False, return_tensors="pt")
        t_ids = t_inputs.pop("input_ids")[0]
        t_mask = t_inputs.pop("attention_mask")[0]
        if "Qwen3VLProcessor" in self.processor.__class__.__name__:
            from ..models.transformers.qwen3_vl import get_rope_index
        else:
            from ..models.transformers.qwen2_vl import get_rope_index
        vision_pos = get_rope_index(
            self.processor, input_ids=t_ids, image_grid_thw=t_inputs.get("image_grid_thw", None),
            video_grid_thw=t_inputs.get("video_grid_thw", None), second_per_grid_ts=t_inputs.get("second_per_grid_ts", None),
            attention_mask=t_mask,
        )
        t_pos = torch.cat((torch.arange(len(t_ids)).unsqueeze(0), vision_pos), dim=0)
        t_ids, t_mask, t_pos = VF.postprocess_data(
            input_ids=t_ids, attention_mask=t_mask, position_ids=t_pos,
            max_length=int(os.environ.get("OPSD_TEACHER_MAX_PROMPT", "6144")),
            pad_token_id=self.tokenizer.pad_token_id, left_pad=True, truncation="error",
        )
        example[f"{prefix}_input_ids"] = t_ids
        example[f"{prefix}_attention_mask"] = t_mask
        example[f"{prefix}_position_ids"] = t_pos
        example[f"{prefix}_multi_modal_data"] = {"videos": q_videos, "images": t_images} if q_videos else {"images": t_images}
        # E20a: the unexpanded prompt ids, so the rollout engine can also generate from this view (future as a hint)
        example[f"{prefix}_raw_prompt_ids"] = self.tokenizer.encode(t_prompt, add_special_tokens=False)
