#train recall
i=0
model_path="model/qwen-14b"
model_version="qwen2_round7_qlora"
lora_path="model_save/${model_version}_rerun/epoch_9_model/adapter.bin"
echo "start"
CUDA_VISIBLE_DEVICES=$i nohup python3 -u recall.py ${model_path} ${model_version} ${lora_path} > log_recall_zero.txt 2>&1


nohup sh run_mistral_cos_argu.sh > log_simcse.txt 2>&1


model_version="qwen2_round9_qlora"
lora_path="model_save/${model_version}_rerun/epoch_9_model/adapter.bin"
CUDA_VISIBLE_DEVICES=$i nohup python3 -u recall.py ${model_path} ${model_version} ${lora_path} > log_recall_train.txt 2>&1


#train rerank
#refer to rank.sh
