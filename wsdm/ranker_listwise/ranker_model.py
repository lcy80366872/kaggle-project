from dataclasses import dataclass
from typing import Optional

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch import Tensor, nn
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.file_utils import ModelOutput


def fixed_cross_entropy(source, target, num_items_in_batch: int = 4, ignore_index: int = -100, **kwargs):
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
    if reduction == "sum":
        loss = loss / num_items_in_batch
    return loss


def ForCausalLMLoss(
    logits, labels,loss_fuc, vocab_size=152064
):
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    shift_labels = shift_labels.view(-1)
    shift_logits = shift_logits.view(-1,  logits.size(-1))
    
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)

    loss_fuc = nn.CrossEntropyLoss(ignore_index=-100)
    
    loss = loss_fuc(shift_logits, shift_labels)
    return loss
def label_smoothed_nll_loss(lprobs, target, eps, num_classes):
    """
    计算标签平滑的负对数似然损失
    lprobs: 模型输出的对数概率，形状 [batch_size, num_classes]
    target: 真实标签（类别索引），形状 [batch_size]
    eps: 平滑因子
    num_classes: 类别总数
    """
    # 创建标签平滑的目标概率分布
    nll_loss = -lprobs.gather(dim=-1, index=target.unsqueeze(-1))  # 标准的交叉熵损失
    smooth_loss = -lprobs.mean(dim=-1)  # 平滑标签的损失

    # 标签平滑: 目标类别的概率从 1.0 调整为 1.0 - eps，其他类别的概率均匀分配
    loss = (1.0 - eps) * nll_loss + eps * smooth_loss  # 标签平滑的最终损失

    return loss.mean()

@dataclass
class RankerOutput(ModelOutput):
    distillation_loss: Optional[Tensor] = None
    ce_loss: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    logits: Optional[Tensor] = None
    causalLMLoss: Optional[Tensor] = None


def get_base_model(cfg):
    config = AutoConfig.from_pretrained(cfg.model.backbone_path, trust_remote_code=cfg.model.trust_remote_code)
    config.use_cache = False
    print("config.is_decoder",config.is_decoder)
    if cfg.model.use_bnb:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=["lm_head"],
        )

        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.backbone_path,
            config=config,
            quantization_config=bnb_config,
            attn_implementation=cfg.model.attn_implementation,
            trust_remote_code=cfg.model.trust_remote_code,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.backbone_path,
            config=config,
            attn_implementation=cfg.model.attn_implementation,
            trust_remote_code=cfg.model.trust_remote_code,
            torch_dtype=torch.bfloat16,
        )
    model.config.pretraining_tp = 1

    # LoRA ---
    if cfg.model.use_lora:
        peft_config = LoraConfig(
            r=cfg.model.lora.r,
            lora_alpha=cfg.model.lora.lora_alpha,
            lora_dropout=cfg.model.lora.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            target_modules=list(cfg.model.lora.target_modules),
            modules_to_save=list(cfg.model.lora.modules_to_save),
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    print(model.config) 
    print("config.is_decoder",model.config.is_decoder)
    return model


class Ranker(nn.Module):
    def __init__(self, cfg, base_model, tokenizer):
        super().__init__()

        self.model = base_model
        self.config = self.model.config
        self.num_labels = cfg.model.num_labels
        self.token_ids = []
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.tok_locations = []
        for letter in letters[: self.num_labels]:
            token_id = tokenizer(letter, add_special_tokens=False)["input_ids"][-1]
            self.tok_locations.append(token_id)

        for idx, letter in enumerate(letters[: self.num_labels]):
            print(f">> Ranker: {letter} token id: {self.tok_locations[idx]}")

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def encode(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        scores = []
        for token_id in self.tok_locations:
            score = outputs.logits[:, -1, token_id]  # [bs]
            scores.append(score)

        logits = torch.stack(scores, 1)  # [bs, num_labels]

        return logits.contiguous()

    def forward(self, input_ids, attention_mask, labels=None,expl_input_ids=None,expl_attention_mask=None,aux_labels=None, teacher_logits=None, ce_mask=None, temperature=1.0, use_distillation=True,multi_task=True, epsilon=0.1, **kwargs):
        logits = self.encode(input_ids, attention_mask)

        loss = None
        distillation_loss = None
        ce_loss = None

        if labels is not None:
            labels = labels.to(logits.device).reshape(-1)  # [bs]
            ce_mask = ce_mask.to(logits.device).reshape(-1)  # [bs]
            ce_loss = (self.loss_fn(logits, labels) * ce_mask).mean()
            aux_loss = torch.tensor(0.0, device=logits.device)
            # 使用标签平滑计算交叉熵损失
            # lprobs = torch.log_softmax(logits, dim=-1)  # 获取对数概率
            # ce_loss = label_smoothed_nll_loss(lprobs, labels, epsilon, num_classes=logits.size(-1)) * ce_mask
            # ce_loss = ce_loss.mean()  # 执行mask和平均

            if use_distillation:
                if teacher_logits is not None:
                    teacher_targets = teacher_logits
                    teacher_targets = torch.softmax(teacher_targets.detach() / temperature, dim=-1)
                    distillation_loss = -torch.mean(torch.sum(torch.log_softmax(logits, dim=-1) * teacher_targets, dim=-1))
                    loss = 0.5 * ce_loss + 0.5 * distillation_loss  # alpha = 0.5, beta = 0.5
                else:
                    raise ValueError("Teacher logits are required for distillation loss")
            else:
                loss = ce_loss
                distillation_loss = torch.tensor(0.0, device=logits.device)
        if teacher_logits is not None:
            teacher_targets = teacher_logits
            teacher_targets = torch.softmax(teacher_targets.detach() / temperature, dim=-1)
            distillation_loss = -torch.mean(torch.sum(torch.log_softmax(logits, dim=-1) * teacher_targets, dim=-1))
            loss =  distillation_loss  # alpha = 0.5, beta = 0.5        
        if multi_task:
            if aux_labels is not None:
                aux_labels = aux_labels.to(logits.device)
                outputs = self.model(input_ids=expl_input_ids, attention_mask=expl_attention_mask, output_hidden_states=True)
                aux_logits =outputs.logits
                aux_loss= ForCausalLMLoss(aux_logits,aux_labels,self.loss_fn,).mean()
                # print("ce-loss",loss)
                loss = 0.5*loss+0.5*aux_loss
            

        return RankerOutput(loss=loss, logits=logits, distillation_loss=distillation_loss, ce_loss=ce_loss,causalLMLoss=aux_loss)

    def save(self, output_dir: str):
        state_dict = self.model.state_dict()
        state_dict = type(state_dict)({k: v.clone().cpu() for k, v in state_dict.items()})
        self.model.save_pretrained(output_dir, state_dict=state_dict)
