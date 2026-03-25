
from collections.abc import Callable
from typing import Optional
import torch
import math

def get_lr_cosine_schedule(t, max_lr, min_lr, T_w, T_c):
    if t < T_w:
        return max_lr * t / T_w
    elif t <= T_c:
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * (t - T_w) / (T_c - T_w)))
    else:
        return min_lr

def gradient_clipping(parameters, max_norm, eps=1e-6):
    parameters = [p for p in parameters if p.grad is not None]
    total_norm = torch.tensor(0.0, device=parameters[0].grad.device)

    for p in parameters:
        total_norm += (p.grad ** 2).sum()

    total_norm = torch.sqrt(total_norm)
    clip_coef = max_norm / (total_norm + eps)
    if clip_coef < 1:
        for p in parameters:
            p.grad.mul_(clip_coef)



class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable[[], float]] = None) -> None: # pyright: ignore[reportIncompatibleMethodOverride]
        if closure is not None:
            closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                state["step"] += 1
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)# m = beta1 * m + (1 - beta1) * g
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)# v = beta2 * v + (1 - beta2) * g^2

                denom = exp_avg_sq.sqrt().add_(group["eps"])# v^0.5 + eps
                bias_correction1 = 1 - beta1 ** state["step"]# 1 - beta1^t
                bias_correction2 = 1 - beta2 ** state["step"]# 1 - beta2^t
                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1# lr * (1 - beta2^t)^0.5 / (1 - beta1^t)
                p.data.mul_(1 - group["lr"] * group["weight_decay"])# p = p * (1 - lr * weight_decay)
                p.data.addcdiv_(exp_avg, denom, value=-step_size)# p = p - step_size * m / (v^0.5 + eps)

