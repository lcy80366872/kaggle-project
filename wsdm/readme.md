# WSDM Cup - Multilingual Chatbot Arena

This repo contains my solution code for the `WSDM Cup - Multilingual Chatbot Arena` Kaggle competition, which won 5st place. The full solution is described [here](https://www.kaggle.com/competitions/wsdm-cup-multilingual-chatbot-arena/discussion/567856). Please refer to the following sections for details on dependencies, training. If you run into any issues with the setup/code or have any questions/suggestions, please feel free to contact me at `chenyulin@mail.nankai.edu.cn`. Thanks!

# 1 Setup
## 1.1 Compute
The models were trained on an instance with the following specifications:
- NVIDIA A100-SXM4-80GB
- RAM: 256 GB
- Disk space: 512 GB

## 1.2 Environment
To train the models,please clone the repo and install the dependencies.

```
git clone https://github.com/rbiswasfc/eedi-mining-misconceptions.git
cd wsdm
conda create -n  wsdm python=3.11
conda activate wsdm
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu118torch2.3cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install -r requirements.txt
```

## 1.3 Download Datasets
Download data in https://www.kaggle.com/datasets/linchenyu/wsdm-5th-2stage-data.
and put them into the "data" directory


## 1.4 Download Models
It is recommended to download the required backbones from HF Hub before training the models.
I use QWEN2.5-14B-Instruct (https://huggingface.co/Qwen/Qwen2.5-14B-Instruct)
Download it and put it into the "model" directory
These backbones will be used for fine-tuning.


# 2. Training
- Data: 
stage1_data(WSDM48k,LMSYS_55K,Add_33k) with hard label
stage2_data(WSDM48k,LMSYS_55K,Add_33k) with soft label  like [0,1]->[0.05,0.95], and  open-model dataset with soft label infer by stage1
The solution pipeline involved two stage train:
- stage1: I trained the Qwen2.5-14b-it model with LoRA in fp16 precision, directly optimizing its output probability distribution for answers A and B based on hard labels, while ignoring the loss for other tokens.
- stage2: Using the trained model of stage1 to perform logits inference on the open-model dataset to obtain the probabilities of answers A and B.Set WSDM48k,LMSYS_55K,Add_33k with soft label like [0,1]->[0.05,0.95]
Training the same as stage 1, just need some adjustment to the loss function.

If you want to track training runs using `wandb`, please log in to your wandb account by running `wandb login` from the terminal.

## 2.1 stage1 train

The models were trained using the `train.py` script. Please run the following commands to fine-tune `Qwen/Qwen2.5-14B-Instruct` for ranker response A and B.

```
accelerate launch train.py --config-name conf_rank_stage1 use_wandb=true
```
## 2.2 stage2 train

After stage1, we will get LoRA adapters,it were then merged with `Qwen/Qwen2.5-14B-Instruct` to create the base model for stage2 logits inferring:
```
python merge_adapter.py \
--backbone_path Qwen/Qwen2.5-14B-Instruct \
--adapter_path output/stage1 \
--save_dir merge_models/qwen14b_stage1
```

Next, I use the merge model to infer soft label of some stage2 data(only open-model dataset):
```
python infer_logits.py
```
we will gain the logits of A and B of open-model dataset.
Besides, we will also change stage1 data, If A is the ground truth, I will set the a_logits column to a specific value and the b_logits column to another specific value, with the aim of making the final softmax probabilities 0.95 and 0.05, respectively.
After that, i will concat these two dataset and gain the stage2 dataset(stage2_data_240k.parquet)

Finally, the similar command as stage 1:
```
accelerate launch train.py --config-name conf_rank_stage2 use_wandb=true
```
we will get the final LoRA adapter and then we merge it:
```
python merge_adapter.py \
--backbone_path Qwen/Qwen2.5-14B-Instruct \
--adapter_path output/stage2 \
--save_dir merge_models/qwen14b_stage2
```
This is the final model.

# 4. Inference
https://www.kaggle.com/code/linchenyu/5th-place-solution
