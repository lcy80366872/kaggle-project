num=2
model_version="qwen2.5_14b_round7_qlora_rerun"
rank_model_path="model/qwen-14b"
rank_lora_path="rank_save/qwen2_round7_qlora_rerun/round3"


python -u get_test_rank_result.py ${model_version} ${rank_model_path} ${num} ${rank_lora_path}
