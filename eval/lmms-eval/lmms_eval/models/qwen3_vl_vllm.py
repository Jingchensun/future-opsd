"""Qwen3-VL served by vLLM, one engine per GPU (data parallel via `accelerate launch`).

Same prompts and visual preprocessing as `qwen3_vl` (system prompt, QWEN3VL_SYSTEM_PROMPT, frame sampling via
qwen_vl_utils, `until` / `<answer>` post-processing), only the generation backend differs, so scores are comparable.

    accelerate launch --num_processes 8 -m lmms_eval --model qwen3_vl_vllm \
        --model_args pretrained=...,max_num_frames=32,extract_answer=True --tasks ... --batch_size 256
`batch_size` is the number of requests handed to vLLM at once (the next chunk is decoded while it generates).
"""
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Union

from accelerate import Accelerator
from loguru import logger as eval_logger
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer

from lmms_eval.api.instance import Instance
from lmms_eval.api.registry import register_model
from lmms_eval.models.qwen3_vl import Qwen3_VL, _strict_bool, process_vision_info

_DIST_ENV = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK", "MASTER_ADDR", "MASTER_PORT",
             "TORCHELASTIC_RUN_ID", "ACCELERATE_USE_FSDP", "ACCELERATE_USE_DEEPSPEED")


@register_model("qwen3_vl_vllm")
class Qwen3_VL_VLLM(Qwen3_VL):
    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3-VL-8B-Instruct",
        batch_size: Optional[Union[int, str]] = 256,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1605632,
        total_pixels: Optional[int] = None,
        max_num_frames: int = 32,
        fps: Optional[float] = None,
        system_prompt: Optional[str] = "You are a helpful assistant.",
        interleave_visuals: Optional[bool] = False,
        enable_thinking: Optional[bool] = None,
        reasoning_prompt: Optional[str] = None,
        extract_answer: bool = False,
        verbose: bool = False,
        gpu_memory_utilization: float = 0.85,
        max_model_len: int = 32768,
        max_num_seqs: int = 128,
        max_images: int = 64,
        max_videos: int = 2,
        preprocess_threads: int = 16,
        reasoning_budget: Optional[int] = None,
        **kwargs,
    ) -> None:
        # skip Qwen3_VL.__init__ (it loads the HF model); set up the same attributes here
        super(Qwen3_VL, self).__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"
        accelerator = Accelerator()
        self.accelerator = accelerator
        self._rank = accelerator.local_process_index if accelerator.num_processes > 1 else 0
        self._world_size = accelerator.num_processes
        self._device = accelerator.device

        self.min_pixels, self.max_pixels, self.total_pixels = min_pixels, max_pixels, total_pixels
        self.max_num_frames, self.fps, self.enable_thinking = max_num_frames, fps, enable_thinking
        self.reasoning_prompt = reasoning_prompt.replace("\\n", "\n") if reasoning_prompt else None
        self.processor = AutoProcessor.from_pretrained(pretrained, max_pixels=max_pixels, min_pixels=min_pixels)
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self.system_prompt = os.environ.get("QWEN3VL_SYSTEM_PROMPT") or system_prompt
        self.extract_answer = _strict_bool(extract_answer)
        self.interleave_visuals = _strict_bool(interleave_visuals)
        self.batch_size_per_gpu = int(batch_size)
        self.preprocess_threads = int(preprocess_threads)
        # budget forcing (extract_answer mode): cap the reasoning at `reasoning_budget` tokens; any output that has no
        # <answer> by then is continued from "...</reason>\n<answer>" for at most 16 tokens, so every item gets an answer
        self.reasoning_budget = int(reasoning_budget) if reasoning_budget not in (None, "", "None") else None
        self.n_forced = 0
        self.verbose = bool(_strict_bool(verbose)) or os.environ.get("QWEN3_VL_VERBOSE", "0") in ("1", "true", "True", "yes")
        self._max_length = max_model_len

        # One engine per rank: the engine process is spawned with only this rank's GPU visible and without the
        # torch.distributed variables of `accelerate launch` (vLLM sets up its own single-GPU group).
        from vllm import LLM

        # torchrun's TORCHELASTIC_USE_AGENT_STORE would make the engine wait for a TCPStore that nobody hosts
        dist_env = _DIST_ENV + tuple(k for k in os.environ if k.startswith("TORCHELASTIC_") and k not in _DIST_ENV)
        cache_env = ("VLLM_CACHE_ROOT", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")
        saved = {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "VLLM_HOST_IP", "VLLM_PORT") + cache_env + dist_env}
        visible = (saved["CUDA_VISIBLE_DEVICES"] or ",".join(map(str, range(64)))).split(",")
        os.environ["CUDA_VISIBLE_DEVICES"] = visible[accelerator.local_process_index] if accelerator.num_processes > 1 else (saved["CUDA_VISIBLE_DEVICES"] or "0")
        for k in dist_env:
            os.environ.pop(k, None)
        os.environ["VLLM_HOST_IP"] = "127.0.0.1"  # the engine's single-GPU group otherwise binds the host's IPv6 address and times out
        # disjoint port ranges per rank: 8 engines picking free ports at the same moment can collide (EADDRINUSE)
        os.environ["VLLM_PORT"] = str(int(os.environ.get("QWEN3VL_VLLM_PORT_BASE", 31000)) + 200 * accelerator.local_process_index)
        # per-rank compile caches: 8 engines compiling at once into one torch.compile / inductor cache can truncate
        # each other's pickles ("pickle data was truncated" -> CompiledFxGraph has no compiled_fn_runner)
        cache_root = os.path.join(os.environ.get("QWEN3VL_VLLM_CACHE", "/tmp/qwen3vl_vllm_cache"), f"rank{accelerator.local_process_index}")
        os.environ["VLLM_CACHE_ROOT"] = os.path.join(cache_root, "vllm")
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(cache_root, "inductor")
        os.environ["TRITON_CACHE_DIR"] = os.path.join(cache_root, "triton")
        try:
            self.llm = LLM(
                model=pretrained,
                dtype="bfloat16",
                gpu_memory_utilization=float(gpu_memory_utilization),
                max_model_len=int(max_model_len),
                max_num_seqs=int(max_num_seqs),
                limit_mm_per_prompt={"image": int(max_images), "video": int(max_videos)},
                mm_processor_kwargs={"min_pixels": min_pixels, "max_pixels": max_pixels},
                seed=0,
                # QWEN3VL_VLLM_EAGER=1: no torch.compile / CUDA graphs. Qwen3-VL-30B-A3B on B200 spent >3 h capturing
                # graphs (~5 min per graph, untuned fused-MoE kernels) before generating anything.
                enforce_eager=os.environ.get("QWEN3VL_VLLM_EAGER", "0") in ("1", "true", "True"),
                # QWEN3VL_VLLM_MM_CACHE_GB=0 turns off the multimodal processor cache: with it on, a request sometimes fails with
                # "Expected a cached item for mm_hash=..." and that rank never returns (E47b_30k SEED L1, 10-31). Unset = vLLM default.
                **({"mm_processor_cache_gb": float(os.environ["QWEN3VL_VLLM_MM_CACHE_GB"])} if os.environ.get("QWEN3VL_VLLM_MM_CACHE_GB") else {}),
            )
        except BaseException:
            # lmms-eval catches model errors and exits 0, which leaves the other ranks waiting in NCCL forever;
            # exit non-zero so torchrun tears the whole group down
            import traceback

            traceback.print_exc()
            eval_logger.error(f"[rank {self._rank}] vLLM engine failed to start, aborting all ranks")
            os._exit(1)
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        eval_logger.info(f"[rank {self._rank}] vLLM engine ready ({pretrained})")

    @property
    def model(self):
        return None

    @property
    def config(self):
        return None

    def _build_messages(self, context, visuals, task_name):
        """Single-request version of Qwen3_VL._preprocess_chunk's message construction."""
        context = context.replace("<image>", "")
        if self.reasoning_prompt:
            context = context.strip() + self.reasoning_prompt
        video_kwargs = self._build_video_kwargs()
        processed = []
        for visual in self.flatten(visuals) if visuals else []:
            if isinstance(visual, str) and visual.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm")):
                processed.append({"type": "video", "video": visual, **self._cap_video_kwargs_to_available_frames(visual, video_kwargs)})
            elif (isinstance(visual, str) and os.path.isfile(visual)) or hasattr(visual, "size"):  # path or PIL image
                processed.append({"type": "image", "image": visual, "max_pixels": self.max_pixels, "min_pixels": self.min_pixels})
        message = [{"role": "system", "content": self._get_system_prompt_for_instance(task_name)}]
        if not self.interleave_visuals:
            message.append({"role": "user", "content": processed + [{"type": "text", "text": context}]})
        else:
            parts, texts = [], re.split(r"<image \d+>", context)
            if texts[0]:
                parts.append({"type": "text", "text": texts[0]})
            for j, ph in enumerate(re.findall(r"<image \d+>", context)):
                k = min(int(re.search(r"\d+", ph).group()) - 1, len(processed) - 1)
                if processed and k >= 0:
                    parts.append(processed[k])
                if j + 1 < len(texts) and texts[j + 1]:
                    parts.append({"type": "text", "text": texts[j + 1]})
            message.append({"role": "user", "content": parts})
        return message, context

    def _prepare(self, args):
        context, gen_kwargs, doc_to_visual, doc_id, task, split = args
        visuals = doc_to_visual(self.task_dict[task][split][doc_id])
        message, context = self._build_messages(context, visuals, task)
        prompt = self._apply_chat_template([message])[0]
        images, videos, video_kwargs = process_vision_info([message], return_video_kwargs=True, image_patch_size=16, return_video_metadata=True)
        mm = {}
        if images:
            mm["image"] = images
        if videos:
            mm["video"] = videos  # [(frames tensor, metadata)], frames already sampled and resized
        req = {"prompt": prompt, "multi_modal_data": mm}
        if videos:
            req["mm_processor_kwargs"] = video_kwargs
        return req, context

    def _sampling_params(self, gen_kwargs):
        from vllm import SamplingParams

        g = {**self.DEFAULT_GEN_KWARGS, **gen_kwargs}
        until = g.get("until", [self.tokenizer.decode(self.eot_token_id)])
        until = [until] if isinstance(until, str) else list(until)
        until = [u for u in until if u and u != "\n\n"]
        greedy = not g.get("temperature") or g.get("do_sample") is False
        # best-of-K (EVAL_NUM_SAMPLES=K > 1): K samples per item at EVAL_TEMPERATURE (default 1.0, as in training)
        k = int(os.environ.get("EVAL_NUM_SAMPLES", "1"))
        if k > 1:
            greedy = False
            g["temperature"], g["top_p"] = float(os.environ.get("EVAL_TEMPERATURE", "1.0")), 1.0
        sp = SamplingParams(
            n=k,
            max_tokens=int(g["max_new_tokens"]),
            temperature=0.0 if greedy else float(g["temperature"]),
            top_p=1.0 if greedy or g.get("top_p") is None else float(g["top_p"]),
            skip_special_tokens=True,
            # per-request seed: runs that differ only in the prompt (E15b hypothesis codes) share draw j's noise unless
            # EVAL_SEED differs (codes 1/2 then gave identical draws 72% of the time vs 48% within a run)
            seed=int(os.environ.get("EVAL_SEED", "0")),
        )
        return sp, until

    def _force_answers(self, items, texts):
        """Second pass for outputs without a closed <answer>: append the end of the reasoning and an opening <answer>
        tag to the prompt and let the model write only the answer."""
        from vllm import SamplingParams

        todo = [j for j, t in enumerate(texts) if not re.search(r"<answer>.*?</answer>", t, re.S)]
        if not todo:
            return texts
        reqs, prefixes = [], []
        for j in todo:
            t = re.sub(r"<answer>.*$", "", texts[j], flags=re.S).rstrip()  # drop an unfinished answer, if any
            tail = ("\n</reason>\n" if "<reason>" in t and "</reason>" not in t else "\n") + "<answer>"
            req = dict(items[j][1])
            req["prompt"] = req["prompt"] + t + tail
            reqs.append(req)
            prefixes.append(t + tail)
        sp = SamplingParams(max_tokens=16, temperature=0.0, stop=["</answer>"], skip_special_tokens=True, seed=0)
        outs = self.llm.generate(reqs, sampling_params=sp, use_tqdm=False)
        texts = list(texts)
        for j, pre, o in zip(todo, prefixes, outs):
            forced = o.outputs[0].text.strip()
            # a model that already stated its answer outside <answer> tags (e.g. the 30B OPSD model, which dropped the
            # tags during training) often closes the forced tag at once; keep its own text then, the task parser reads it
            if forced:
                texts[j] = pre + forced + "</answer>"
        self.n_forced += len(todo)
        eval_logger.info(f"[rank {self.rank}] budget forcing: {len(todo)} of {len(items)} outputs needed a forced answer (total {self.n_forced})")
        return texts

    def generate_until(self, requests: List[Instance]) -> List[str]:
        args = [r.args for r in requests]
        n, bs = len(args), self.batch_size_per_gpu
        res = [None] * n
        pbar = tqdm(total=n, disable=(self.rank != 0), desc="Model Responding")
        pool = ThreadPoolExecutor(max_workers=self.preprocess_threads)
        dump = None
        if os.environ.get("EVAL_SAMPLES_DUMP") and int(os.environ.get("EVAL_NUM_SAMPLES", "1")) > 1:
            os.makedirs(os.environ["EVAL_SAMPLES_DUMP"], exist_ok=True)
            task = requests[0].task_name if requests else "none"
            dump = open(os.path.join(os.environ["EVAL_SAMPLES_DUMP"], f"{task}.rank{self.rank}.jsonl"), "a")
        raw_dump = None  # EVAL_RAW_DUMP=<dir>: every rank writes the full generated text of sample 0 (e.g. for case studies)
        if os.environ.get("EVAL_RAW_DUMP"):
            os.makedirs(os.environ["EVAL_RAW_DUMP"], exist_ok=True)
            task = requests[0].task_name if requests else "none"
            raw_dump = open(os.path.join(os.environ["EVAL_RAW_DUMP"], f"{task}.rank{self.rank}.jsonl"), "a")
        chunks = [list(range(i, min(i + bs, n))) for i in range(0, n, bs)]
        submit = lambda c: [pool.submit(self._prepare, args[i]) for i in c]
        pending = submit(chunks[0]) if chunks else []
        for ci, chunk in enumerate(chunks):
            prepared = [f.result() for f in pending]
            if ci + 1 < len(chunks):
                pending = submit(chunks[ci + 1])  # decode the next chunk's videos while this one generates
            groups = {}  # requests in one chunk may carry different gen_kwargs
            for i, (req, ctx) in zip(chunk, prepared):
                sp, until = self._sampling_params(args[i][1])
                groups.setdefault(repr((sp, until)), (sp, until, []))[2].append((i, req, ctx))
            for sp, until, items in groups.values():
                if self.reasoning_budget and self.extract_answer:
                    sp.max_tokens = min(sp.max_tokens, self.reasoning_budget)
                outs = self.llm.generate([r for _, r, _ in items], sampling_params=sp, use_tqdm=False)
                # item-major flat list of the sp.n samples per item; budget forcing treats each sample on its own
                flat = [(it, c.text) for it, o in zip(items, outs) for c in o.outputs]
                texts = [t for _, t in flat]
                if self.reasoning_budget and self.extract_answer:
                    texts = self._force_answers([it for it, _ in flat], texts)
                answers = []
                for raw in texts:
                    ans = raw
                    for term in until:
                        ans = ans.split(term)[0]
                    ans = self._strip_thinking(ans)
                    if self.extract_answer:
                        m = re.search(r"<answer>(.*?)</answer>", ans, re.S)
                        ans = m.group(1).strip() if m else ans.strip()
                    answers.append(ans)
                for j, (i, _, ctx) in enumerate(items):
                    ans, raw = answers[j * sp.n], texts[j * sp.n]
                    if self.verbose and self.rank == 0:
                        print("\033[33m[qwen3_vl_vllm][raw   ]\033[0m", raw)
                        print("\033[32m[qwen3_vl_vllm][answer]\033[0m", ans, flush=True)
                    res[i] = ans  # lmms-eval scores sample 0; all sp.n samples go to EVAL_SAMPLES_DUMP
                    self.cache_hook.add_partial("generate_until", (ctx, args[i][1]), ans)
                    if raw_dump:
                        r = requests[i]
                        raw_dump.write(json.dumps({"task": r.task_name, "doc_id": r.doc_id, "raw": raw, "answer": ans}) + "\n")
                        raw_dump.flush()
                    if sp.n > 1 and os.environ.get("EVAL_SAMPLES_DUMP"):
                        r = requests[i]
                        dump.write(json.dumps({"task": r.task_name, "doc_id": r.doc_id,
                                               "answers": answers[j * sp.n:(j + 1) * sp.n]}) + "\n")
                        dump.flush()
            pbar.update(len(chunk))
        pool.shutdown()
        pbar.close()
        if dump:
            dump.close()
        if raw_dump:
            raw_dump.close()
        return res
