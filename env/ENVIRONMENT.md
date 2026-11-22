# Environments

All paper results were produced on **8 × NVIDIA B200 (sm_100)** per job (Debian 12, system Python 3.11.2, virtualenvs,
no conda). Each environment below is rebuilt with `bash env/install_env.sh <env>` from `env/lock/<env>.txt`, the
`pip freeze --all` of the environment the runs used.

| env (`$ENV_ROOT/<env>`) | used for | torch / CUDA | key packages |
|-|-|-|-|
| `future-l1-rl-vllm011` | **Qwen3-VL training** (Naive OPSD, Future-OPSD, TTRL; `train/rl`), FSDP → HF merge (`train/rl/tools/model_merger.py`) | 2.8.0 / 12.8 | vLLM 0.11.0 (the custom GPU model runner of `train/rl` only works with 0.11.0), transformers 4.57.1, flash-attn 2.8.3, ray 2.58.0 |
| `future-l1-rl` | **Qwen3-VL evaluation** (lmms-eval, vLLM backend: TwiFF-Bench, SEED-Bench-R1, TempCompass, TemporalBench, Video-MME, MVBench) | 2.9.1 / 12.8 | vLLM 0.16.0, transformers 4.57.1, flash-attn 2.8.3 (compiled), lmms-eval (editable, `eval/lmms-eval`) |
| `future-l1-q35` | **Qwen3.5** training (`train/rl_qwen35`), SFT and evaluation | 2.13.0 / 13.0 | vLLM 0.30.0, transformers 5.17.0, deepspeed 0.18.4, lmms-eval (editable) |
| `judge-vllm` | Qwen3.6-27B **judge server** for TwiFF-Bench (`eval/eval_model.sh` starts it) | 2.13.0 / 13.0 | vLLM 0.30.0, transformers 5.17.0 |
| `future-l1-sft` | **Qwen3-VL SFT** baseline (`train/sft`, deepspeed ZeRO-2) | 2.9.1 / 12.8 | transformers 4.57.1, deepspeed 0.18.2, flash-attn 2.8.3 (compiled), lmms-eval (editable) |

Notes
- B200 needs CUDA 12.8+ builds of torch (sm_100 kernels); older wheels such as torch 2.6 / cu124 do not run on it.
- flash-attn 2.8.3 has no official wheel for torch 2.9: `install_env.sh` compiles it for the local GPUs (or `FA_ARCHS`).
  On a host with a memory limit keep `MAX_JOBS` low (each `cicc` takes about 5 GB); the torch CUDA version must match
  the local `nvcc`.
- HF Qwen-VL's Conv3d patch embedding is very slow under torch 2.9; `train/sft/src/fast_patch_embed.py` replaces it with
  an equivalent linear layer.
- Evaluation runs offline: `HF_HOME=$HF_HOME_EVAL HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` (set by `eval/*.sh`).
