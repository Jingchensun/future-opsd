#!/bin/bash
# Turn an SFT output folder (train/train_sft_text.sh) into a folder that vLLM / transformers can load: <SFT_DIR>/hf_eval.
# The SFT trainer saves its own model class name in config.json, which vLLM does not know.
#   Qwen3-VL : symlink the SFT weights and tokenizer / processor files, rewrite config.json with
#              architectures = [Qwen3VLForConditionalGeneration]
#   Qwen3.5  : symlink the SFT weights; config, chat template, tokenizer and processor files come from the base model
#
#   bash utils/make_hf_eval.sh <SFT_DIR> [BASE_DIR, Qwen3.5 only; default $CKPT_ROOT/Qwen3.5-9B]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source $ROOT/config/paths.sh
O=$(cd "${1:?SFT output dir}" && pwd); E=$O/hf_eval
MT=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('model_type',''))" "$O/config.json")
rm -rf "$E"; mkdir -p "$E"
link(){ for f in "$@"; do if [ -e "$f" ]; then ln -sf "$f" "$E/"; fi; done; }
case $MT in
  qwen3_vl)
    link $O/*.safetensors $O/model.safetensors.index.json $O/added_tokens.json $O/special_tokens_map.json \
         $O/tokenizer.json $O/tokenizer_config.json $O/vocab.json $O/merges.txt $O/chat_template.jinja \
         $O/generation_config.json $O/preprocessor_config.json $O/video_preprocessor_config.json
    python3 - "$O/config.json" "$E/config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1])); c["architectures"] = ["Qwen3VLForConditionalGeneration"]
json.dump(c, open(sys.argv[2], "w"), indent=1)
PY
    ;;
  qwen3_5)
    B=$(cd "${2:-$CKPT_ROOT/Qwen3.5-9B}" && pwd)
    link $O/*.safetensors $O/model.safetensors.index.json
    link $B/config.json $B/chat_template.jinja $B/tokenizer.json $B/tokenizer_config.json $B/vocab.json $B/merges.txt \
         $B/preprocessor_config.json $B/video_preprocessor_config.json
    ;;
  *) echo "unsupported model_type '$MT' in $O/config.json"; exit 1;;
esac
echo "ready: $E ($MT)"
