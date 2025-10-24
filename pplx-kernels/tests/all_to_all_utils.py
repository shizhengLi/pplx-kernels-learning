import dataclasses

import torch


@dataclasses.dataclass
class MoEConfig:
    num_experts: int
    experts_per_token: int
    hidden_dim: int
    max_num_tokens: int
    in_dtype: torch.dtype = torch.bfloat16
    out_dtype: torch.dtype = torch.bfloat16
    block_size: int = 128


@dataclasses.dataclass
class RankTestData:
    def __init__(
        self,
        cfg: MoEConfig,
        rng: torch.Generator,
        use_max_tokens: bool,
    ) -> None:
        self.num_tokens = (
            int(torch.randint(1, cfg.max_num_tokens, [1], generator=rng).item())
            if not use_max_tokens
            else cfg.max_num_tokens
        )
        self.indices = torch.empty(
            self.num_tokens,
            cfg.experts_per_token,
            dtype=torch.int32,
        )
        for i in range(self.num_tokens):
            perm = torch.randperm(cfg.num_experts, generator=rng)
            self.indices[i] = perm[: cfg.experts_per_token]
        self.weights = torch.rand(
            self.num_tokens, cfg.experts_per_token, dtype=torch.float32, generator=rng
        )
        self.x_scale: torch.Tensor | None = None
        if cfg.in_dtype.itemsize == 1:
            x_fp32 = torch.rand(
                self.num_tokens, cfg.hidden_dim, dtype=torch.float32, generator=rng
            )
            self.x = ((x_fp32 - 0.5) * 400).to(cfg.in_dtype)
            self.x_scale = torch.rand(
                self.num_tokens,
                (cfg.hidden_dim + cfg.block_size - 1) // cfg.block_size,
                dtype=torch.float32,
                generator=rng,
            )
        else:
            self.x = torch.randn(
                self.num_tokens, cfg.hidden_dim, dtype=cfg.in_dtype, generator=rng
            )
