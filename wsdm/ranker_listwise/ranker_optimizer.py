import bitsandbytes as bnb
from torch import optim


def get_optimizer_grouped_parameters(cfg, model, print_fn=print):
    param_dict = {name: param for name, param in model.named_parameters()}
    param_dict = {name: param for name, param in param_dict.items() if param.requires_grad}

    # param shape based groupings --
    params_dict_no_decay = {name: param for name, param in param_dict.items() if len(param.shape) == 1}
    params_dict_decay = {name: param for name, param in param_dict.items() if len(param.shape) != 1}

    params_dict_lora_a = {name: param for name, param in params_dict_decay.items() if "lora_A" in name}
    params_dict_lora_b = {name: param for name, param in params_dict_decay.items() if "lora_B" in name}
    params_dict_embed_tokens = {name: param for name, param in params_dict_decay.items() if "embed_tokens" in name}
    params_dict_remaining = {name: param for name, param in params_dict_decay.items() if not any(x in name for x in ["lora_A", "lora_B", "embed_tokens"])}

    # info ---
    def print_param_group_info(group, group_name):
        n_params = round(sum(p.numel() for p in group.values()) / 1e6, 2)
        print_fn(f"{group_name}: # params: {n_params}M | Sample keys: {list(group.keys())[:2]}")

    # print info for each parameter group
    print_param_group_info(params_dict_no_decay, "Optimizer (no_decay)")
    print_param_group_info(params_dict_lora_a, "Optimizer (lora_a)")
    print_param_group_info(params_dict_lora_b, "Optimizer (lora_b)")
    print_param_group_info(params_dict_embed_tokens, "Optimizer (embed_tokens)")
    print_param_group_info(params_dict_remaining, "Optimizer (remaining)")

    # create optimizer groups ---
    wd = cfg.optimizer.weight_decay
    optim_groups = [
        {"params": list(params_dict_no_decay.values()), "lr": cfg.optimizer.lr, "weight_decay": 0.0},
        {"params": list(params_dict_lora_a.values()), "lr": cfg.optimizer.lr_lora_a, "weight_decay": wd},
        {"params": list(params_dict_lora_b.values()), "lr": cfg.optimizer.lr_lora_b, "weight_decay": wd},
        {"params": list(params_dict_embed_tokens.values()), "lr": cfg.optimizer.lr_embed_tokens, "weight_decay": wd},
        {"params": list(params_dict_remaining.values()), "lr": cfg.optimizer.lr, "weight_decay": wd},
    ]

    # filter out groups with no params
    optim_groups = [group for group in optim_groups if len(group["params"]) > 0]
    return optim_groups


def get_optimizer(cfg, model, print_fn=print):
    _optimizers = {
        "Adam": optim.Adam,
        "AdamW": optim.AdamW,
        "AdamW8bit": bnb.optim.Adam8bit,
    }
    assert cfg.optimizer.name in _optimizers, f"Optimizer {cfg.optimizer.name} not supported"

    optim_groups = get_optimizer_grouped_parameters(cfg, model, print_fn)

    optimizer = _optimizers[cfg.optimizer.name](
        optim_groups,
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        betas=(cfg.optimizer.adam_beta_1, cfg.optimizer.adam_beta_2),
        eps=cfg.optimizer.adam_epsilon,
    )

    return optimizer
