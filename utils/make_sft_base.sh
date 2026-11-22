#!/bin/bash
# SFT base = the HF base model + the repo chat template (config/chat_template.json), as $CKPT_ROOT/<model>-futurel1:
# every file of the base is symlinked, chat_template.json is copied from config/.
#   bash utils/make_sft_base.sh Qwen3-VL-8B-Instruct      -> $CKPT_ROOT/Qwen3-VL-8B-Instruct-futurel1
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source $ROOT/config/paths.sh
B=$CKPT_ROOT/${1:?base model dir name under CKPT_ROOT}; O=$B-futurel1
[ -f $B/config.json ] || { echo "no model in $B"; exit 1; }
mkdir -p $O
for f in $B/*; do [ "$(basename $f)" = chat_template.json ] || ln -sfn $f $O/; done
cp $ROOT/config/chat_template.json $O/chat_template.json
echo "made $O"
