#!/bin/bash
#!/bin/bash



PATH_PRE="./output/"
DATA_NAME="qwen2_round7_qlora_recall_top_100_train.jsonl"
DATA_DIR=${PATH_PRE}
MODEL_USE="qwen2_round9"
ZERO_STAGE=2
OUTPUT=./model_save/${MODEL_USE}_qlora_rerun


# 模型地址
MODEL_PATH="model/qwen-14b"
mkdir -p ${OUTPUT}
echo ${ZERO_STAGE}
echo ${OUTPUT}

MASTER_PORT=$(shuf -n 1 -i 10000-65535)
echo ${MASTER_PORT}
deepspeed  --master_port ${MASTER_PORT}  --include localhost:0,1 simcse_deepspeed_mistrial_qlora_argu.py \
       --project_name ${name}_${MODEL_USE} \
       --train_data ${DATA_DIR}${DATA_NAME} \
       --model_name_or_path ${MODEL_PATH} \
       --per_device_train_batch_size 4 \
       --per_device_eval_batch_size 4 \
       --train_group_size 8 \
       --gradient_accumulation_steps 8 \
       --query_max_len 512 \
       --passage_max_len 512 \
       --earystop 0 \
       --save_batch_steps 100000000000 \
       --eary_stop_epoch 5 \
       --save_per_epoch 1 \
       --num_train_epochs 10 \
       --learning_rate 1e-4 \
       --num_warmup_steps 100 \
       --weight_decay 0.01 \
       --lr_scheduler_type cosine \
       --seed 1234 \
       --zero_stage $ZERO_STAGE \
       --deepspeed \
       --output_dir $OUTPUT \
       --gradient_checkpointing