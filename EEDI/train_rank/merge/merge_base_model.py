from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    BitsAndBytesConfig
)

def merge_llm(model_name_or_path, lora_name_or_path, save_path, cache_dir: str = None, token: str = None):
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                 quantization_config=bnb_config,
                                                 cache_dir=cache_dir,
                                                 token=token,
                                                 trust_remote_code=True)
    model = PeftModel.from_pretrained(model, lora_name_or_path)
    model = model.merge_and_unload()
    model.save_pretrained(save_path)

    try:
        tokenizer = AutoTokenizer.from_pretrained(lora_name_or_path)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                                  cache_dir=cache_dir,
                                                  token=token,
                                                  trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            if tokenizer.unk_token_id is not None:
                tokenizer.pad_token_id = tokenizer.unk_token_id
            elif tokenizer.eod_id is not None:
                tokenizer.pad_token_id = tokenizer.eod_id
                tokenizer.bos_token_id = tokenizer.im_start_id
                tokenizer.eos_token_id = tokenizer.im_end_id
        if 'mistral' in model_name_or_path.lower():
            tokenizer.padding_side = 'left'
    tokenizer.save_pretrained(save_path)
if __name__ == "__main__":
    merge_llm('google/gemma-2b', 'lora_llm_output_path', 'merged_model_output_paths')