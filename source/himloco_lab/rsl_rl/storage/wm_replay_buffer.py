# Copyright (c) 2025, HimLoco Lab Project Developers.
# Adapted from SuperMimic (supermimic/core/replay.py)

from __future__ import annotations
import torch


class WMReplayBuffer:
    """Time-major GPU ring replay buffer for SuperMimic World Model & BPTT imagination.
    Windows never cross episode reset boundaries.
    """

    def __init__(self, capacity: int, num_envs: int, device: str | torch.device = "cuda:0"):
        if capacity <= 0 or num_envs <= 0:
            raise ValueError("Replay dimensions must be positive")
        self.capacity = int(capacity)
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.storage: dict[str, torch.Tensor] = {}
        self.ptr = 0
        self.size = 0
        self.total_steps = 0
        self.episode_ids = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    @torch.no_grad()
    def add(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        action: torch.Tensor,
        prev_action: torch.Tensor,
        pre_pre_action: torch.Tensor,
        obs_history: torch.Tensor,
        next_obs_history: torch.Tensor,
        command: torch.Tensor,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        feet_contact: torch.Tensor | None = None,
    ):
        terminated = terminated.bool().view(self.num_envs)
        truncated = truncated.bool().view(self.num_envs)
        values = {
            "state": state,
            "next_state": next_state,
            "action": action,
            "prev_action": prev_action,
            "pre_pre_action": pre_pre_action,
            "obs_history": obs_history,
            "next_obs_history": next_obs_history,
            "command": command,
            "reward": reward.view(self.num_envs),
            "terminated": terminated,
            "truncated": truncated,
            "episode_id": self.episode_ids.clone(),
        }
        if feet_contact is not None:
            values["feet_contact"] = feet_contact

        if not self.storage:
            self.storage = {
                k: torch.empty((self.capacity, *v.shape), dtype=v.dtype, device=self.device)
                for k, v in values.items()
            }

        for k, v in values.items():
            self.storage[k][self.ptr].copy_(v.detach().to(self.device))

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)
        self.total_steps += 1
        self.episode_ids += (terminated | truncated).long()

    def rows(self, ages: torch.Tensor) -> torch.Tensor:
        return (self.ptr - self.size + ages) % self.capacity

    @torch.no_grad()
    def sample_prefix(
        self,
        batch_size: int,
        horizon: int = 8,
        env_ids: list[int] | torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """采样长度为 horizon (默认 8 步) 的连续序列，并自动屏蔽 Reset 之后的所有步"""
        if self.size < horizon:
            raise ValueError(f"Not enough replay steps ({self.size}) for horizon ({horizon})")

        if env_ids is None:
            envs_pool = torch.arange(self.num_envs, device=self.device)
        else:
            envs_pool = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        max_age = self.size - horizon
        ages = torch.randint(0, max_age + 1, (batch_size,), device=self.device)
        env_choices = torch.randint(0, len(envs_pool), (batch_size,), device=self.device)
        envs = envs_pool[env_choices]

        step_rows = self.rows(ages[:, None] + torch.arange(horizon, device=self.device))
        out = {k: v[step_rows, envs[:, None]] for k, v in self.storage.items()}

        done = out["terminated"] | out["truncated"]
        episodes = out["episode_id"]
        preceding = torch.cat((torch.ones_like(done[:, :1]), ~done[:, :-1]), dim=1)
        mask = (preceding & (episodes == episodes[:, :1])).long().cumprod(dim=1).bool()
        out["_mask"] = mask

        # 将 mask=False 的无效步替换为序列起始步，防止脏物理状态产生 NaN 梯度
        for k in list(out.keys()):
            if k.startswith("_"):
                continue
            shape = (*mask.shape, *((1,) * (out[k].ndim - 2)))
            out[k] = torch.where(mask.reshape(shape), out[k], out[k][:, :1])

        return out

    @torch.no_grad()
    def sample_starts(
        self,
        batch_size: int,
        env_ids: list[int] | torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """采样用于 BPTT 想象滚动的真实初始状态（排除已触发终止的无效帧）"""
        if env_ids is None:
            envs_pool = torch.arange(self.num_envs, device=self.device)
        else:
            envs_pool = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        ages = torch.arange(self.size, device=self.device)
        rows = self.rows(ages)
        done = (self.storage["terminated"] | self.storage["truncated"])[rows][:, envs_pool]
        valid_age_idx, valid_env_col = (~done).nonzero(as_tuple=True)

        if len(valid_age_idx) == 0:
            valid_age_idx = torch.randint(0, self.size, (batch_size,), device=self.device)
            valid_env_col = torch.randint(0, len(envs_pool), (batch_size,), device=self.device)

        picks = torch.randint(0, len(valid_age_idx), (batch_size,), device=self.device)
        chosen_rows = self.rows(valid_age_idx[picks])
        chosen_envs = envs_pool[valid_env_col[picks]]

        keys = ("state", "obs_history", "action", "prev_action", "pre_pre_action", "command")
        return {k: self.storage[k][chosen_rows, chosen_envs].clone() for k in keys if k in self.storage}