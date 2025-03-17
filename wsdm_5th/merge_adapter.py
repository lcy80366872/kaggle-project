import argparse

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer


def merge_adapter(backbone_path: str, adapter_path: str, save_dir: str, causal_lm: bool) -> None:
    config = AutoConfig.from_pretrained(backbone_path)
    config.use_cache = False

    if causal_lm:
        ModelCls = AutoModelForCausalLM
    else:
        ModelCls = AutoModel

    model = ModelCls.from_pretrained(backbone_path, config=config, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, device_map="auto")

    model.config.pretraining_tp = 1
    tokenizer = AutoTokenizer.from_pretrained(backbone_path)

    model = PeftModel.from_pretrained(model, adapter_path)
    merged_model = model.merge_and_unload(safe_merge=True)

    merged_model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone_path", type=str, required=True, help="Path to backbone model")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to LoRA adapter")
    parser.add_argument("--save_dir", type=str, required=True, help="Path to save the merged model")
    parser.add_argument("--causal_lm", type=bool, required=True, help="Whether the model is causal LM")

    args = parser.parse_args()

    merge_adapter(args.backbone_path, args.adapter_path, args.save_dir, args.causal_lm)

    # Usage:
    # python merge_adapter.py --backbone_path  model/qwen-14b --adapter_path output/qwen_listwise_14b --save_dir merge_model/merge_qwen --causal_lm False
    # python merge_adapter.py --backbone_path ../models/qwen_listwise_awq --adapter_path ../models/listwise_awq --save_dir ../models/qwen_listwise_merged --causal_lm False
    # python merge_adapter.py --backbone_path ../models/qwen_reasoner_awq --adapter_path ../models/reasoner_awq --save_dir ../models/qwen_reasoner_merged --causal_lm True
