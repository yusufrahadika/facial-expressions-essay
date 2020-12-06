import math

import torch
from torch.optim.optimizer import Optimizer, required
from .gc import centralized_gradient


class RAdam(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        N_sma_threshhold=5,
        use_gc=True,
        gc_conv_only=False,
        gc_loc=True,
        diffgrad=True,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        buffer = [None, None, None]

        if (
            isinstance(params, (list, tuple))
            and len(params) > 0
            and isinstance(params[0], dict)
        ):
            for param in params:
                if "betas" in param and (
                    param["betas"][0] != betas[0] or param["betas"][1] != betas[1]
                ):
                    param["buffer"] = buffer
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            buffer=buffer,
            N_sma_threshhold=N_sma_threshhold,
        )
        super().__init__(params, defaults)

        # gc on or off
        self.gc_loc = gc_loc
        self.use_gc = use_gc
        self.gc_conv_only = gc_conv_only

        # diffgrad
        self.diffgrad = diffgrad

        print(
            f"RAdam optimizer loaded. \nGradient Centralization usage = {self.use_gc} \nDiffgrad usage = {self.diffgrad}"
        )
        if self.use_gc and self.gc_conv_only == False:
            print(f"GC applied to both conv and fc layers")
        elif self.use_gc and self.gc_conv_only == True:
            print(f"GC applied to conv layers only")

    def step(self, closure=None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()
                if grad.is_sparse:
                    raise RuntimeError("RAdam does not support sparse gradients")

                p_data_fp32 = p.data.float()

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p_data_fp32)
                    state["exp_avg_sq"] = torch.zeros_like(p_data_fp32)
                    state["previous_grad"] = torch.zeros_like(p.data)
                else:
                    state["exp_avg"] = state["exp_avg"].type_as(p_data_fp32)
                    state["exp_avg_sq"] = state["exp_avg_sq"].type_as(p_data_fp32)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                previous_grad = state["previous_grad"]
                beta1, beta2 = group["betas"]

                # GC operation for Conv layers and FC layers
                if self.gc_loc:
                    grad = centralized_gradient(
                        grad, use_gc=self.use_gc, gc_conv_only=self.gc_conv_only
                    )

                # Decay the first and second moment running average coefficient
                # m_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # v_t
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                state["step"] += 1
                buffered = group["buffer"]
                if state["step"] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state["step"]
                    beta1_t = math.pow(beta1, state["step"])
                    beta2_t = math.pow(beta2, state["step"])
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * state["step"] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma

                    # more conservative since it's an approximated value
                    if N_sma >= 5:
                        step_size = math.sqrt(
                            (1 - beta2_t)
                            * (N_sma - 4)
                            / (N_sma_max - 4)
                            * (N_sma - 2)
                            / N_sma
                            * N_sma_max
                            / (N_sma_max - 2)
                        )
                    else:
                        step_size = 1.0
                    step_size /= 1 - beta1_t
                    buffered[2] = step_size

                if group["weight_decay"] != 0:
                    p_data_fp32.add_(
                        p_data_fp32, alpha=-group["weight_decay"] * group["lr"]
                    )

                # GC operation
                if self.gc_loc == False:
                    G_grad = centralized_gradient(
                        G_grad, use_gc=self.use_gc, gc_conv_only=self.gc_conv_only
                    )

                # Ignore this variable if not using DiffGrad
                friction = 1.0
                if self.diffgrad:
                    diff = torch.norm(previous_grad - grad)
                    # Calculate DiffGrad Friction Coefficient
                    friction = abs(torch.sigmoid(diff))
                    state["previous_grad"] = grad.clone()

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])
                    p_data_fp32.addcdiv_(
                        exp_avg, denom, value=-step_size * friction * group["lr"]
                    )
                else:
                    p_data_fp32.add_(exp_avg, alpha=-step_size * friction * group["lr"])

                p.data.copy_(p_data_fp32)

        return loss
