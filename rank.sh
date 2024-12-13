
# --from_peft "rank_save/qwen2.5_1.5b_round7_qlora_rerun/checkpoint-531" \
PATH_PRE="./output/"
DATA_DIR=${PATH_PRE}
DATA_NAME="last_rank_train.jsonl"
# DATA_NAME="test0.jsonl"
MODEL_USE="qwen2.5_14b_round7"
OUTPUT=./rank_save/${MODEL_USE}_qlora_rerun
MODEL_PATH="model/qwen-14b"
mkdir -p ${OUTPUT}
nohup torchrun --nproc_per_node 5 \
-m train_rank.run \
--output_dir ${OUTPUT} \
--model_name_or_path ${MODEL_PATH} \
--train_data ${DATA_DIR}${DATA_NAME} \
--learning_rate 2e-4 \
--num_train_epochs 2 \
--per_device_train_batch_size 4 \
--gradient_accumulation_steps 4 \
--dataloader_drop_last True \
--query_max_len 256 \
--passage_max_len 256 \
--train_group_size 16 \
--logging_steps 1 \
--save_steps 111111 \
--save_total_limit 50 \
--ddp_find_unused_parameters False \
--gradient_checkpointing \
--deepspeed stage1.json \
--warmup_ratio 0.1 \
--bf16 \
--use_lora True \
--lora_rank 32 \
--lora_alpha 64 \
--use_flash_attn False \
--target_modules q_proj k_proj v_proj o_proj down_proj up_proj gate_proj > log_rank.txt 2>&1