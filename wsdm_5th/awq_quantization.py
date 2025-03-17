import argparse
from datasets import Dataset

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, AwqConfig
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

    # merged_model.save_pretrained(save_dir)
    # tokenizer.save_pretrained(save_dir)
    return merged_model,tokenizer
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="../models/qwen_pointwise_32b_merged")
    parser.add_argument("--quant_path", type=str, default="../models/qwen_pointwise_32b_merged_awq")
    parser.add_argument("--calib_data", type=str, default="rbiswasfc/eedi-awq-calibration-oracle-new")
    parser.add_argument("--max_calib_seq_len", type=int, default=1024)
    args = parser.parse_args()

    model_path = args.model_path
    quant_path = args.quant_path
    calib_data = args.calib_data
    max_calib_seq_len = args.max_calib_seq_len

    quant_config = {"zero_point": True, "q_group_size": 64, "w_bit": 4, "version": "GEMM"}

    # Load model
    model = AutoAWQForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    calib_data_file= Dataset.from_parquet(calib_data)
    calib_data_file=calib_data_file['text']
    print(calib_data_file[0])
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data_file, max_calib_seq_len=max_calib_seq_len)

    quantization_config = AwqConfig(
        bits=quant_config["w_bit"],
        group_size=quant_config["q_group_size"],
        zero_point=quant_config["zero_point"],
        version=quant_config["version"].lower(),
    ).to_dict()

    model.model.config.quantization_config = quantization_config
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)

# Usage:
# python awq_quantization.py --model_path awq_model/merge_qwen --quant_path awq_models/awq_qwen --calib_data output/calib_data.parquet --max_calib_seq_len 1024
# python awq_quantization.py --model_path ../models/qwen_listwise_merged --quant_path ../models/listwise_awq --calib_data rbiswasfc/eedi-awq-calibration-tutor --max_calib_seq_len 1600
# python awq_quantization.py --model_path ../models/qwen_reasoner_merged --quant_path ../models/reasoner_awq --calib_data rbiswasfc/eedi-awq-calibration-cot --max_calib_seq_len 1024
