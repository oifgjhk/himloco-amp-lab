# source/himloco_lab/rsl_rl/storage/wm_replay_buffer.py
from __future__ import annotations
import torch


class WorldModelReplayBuffer:
    """SuperMimic 环形经验回放池 B (1000 steps x 4096 envs，常驻 GPU 显存)"""

    def __init__(
        self,
        num_envs: int,
        capacity_per_env: int,
        state_dim: int,
        action_dim: int,
        obs_dim: int,
        obs_hist_len: int,
        cmd_dim: int,
        context_dim: int,
        device: str = "cuda",
    ):
        self.num_envs = num_envs
        self.capacity = capacity_per_env
        self.device = device
        self.ptr = 0
        self.size = 0

        self.states = torch.zeros(capacity_per_env, num_envs, state_dim, device=device)
        self.next_states = torch.zeros(capacity_per_env, num_envs, state_dim, device=device)
        self.actions = torch.zeros(capacity_per_env, num_envs, action_dim, device=device)
        self.clean_obs = torch.zeros(capacity_per_env, num_envs, obs_dim, device=device)
        self.next_clean_obs = torch.zeros(capacity_per_env, num_envs, obs_dim, device=device)
        self.obs_history = torch.zeros(capacity_per_env, num_envs, obs_hist_len, obs_dim, device=device)
        self.rewards = torch.zeros(capacity_per_env, num_envs, device=device)
        self.dones = torch.zeros(capacity_per_env, num_envs, dtype=torch.bool, device=device)
        self.commands = torch.zeros(capacity_per_env, num_envs, cmd_dim, device=device)
        self.contexts = torch.zeros(capacity_per_env, num_envs, context_dim, device=device)

    @torch.no_grad()
    def insert(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        action: torch.Tensor,
        clean_obs: torch.Tensor,
        next_clean_obs: torch.Tensor,
        obs_hist: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        command: torch.Tensor,
        context: torch.Tensor,
    ):
        self.states[self.ptr].copy_(state)
        self.next_states[self.ptr].copy_(next_state)
        self.actions[self.ptr].copy_(action)
        self.clean_obs[self.ptr].copy_(clean_obs)
        self.next_clean_obs[self.ptr].copy_(next_clean_obs)
        self.obs_history[self.ptr].copy_(obs_hist)
        self.rewards[self.ptr].copy_(reward)
        self.dones[self.ptr].copy_(done.bool())
        self.commands[self.ptr].copy_(command)
        self.contexts[self.ptr].copy_(context)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_k_step_windows(self, batch_size: int = 2048, k: int = 8):
        """采样 k=8 步连续轨迹窗口，自动过滤跨 Episode Reset 或跨环形写入指针 ptr 的窗口"""
        max_start = self.size - k
        assert max_start > 0, "Replay buffer 数据量尚不足 k 步！"

        # 随机采样候选 (t_start, env_id)
        t_starts = torch.randint(0, max_start, (batch_size * 2,), device=self.device)
        env_ids = torch.randint(0, self.num_envs, (batch_size * 2,), device=self.device)

        # 构建时间步索引矩阵 [2*B, k]
        steps = t_starts.unsqueeze(1) + torch.arange(k, device=self.device).unsqueeze(0)
        # 过滤条件 1：前 k-1 步内不能包含 done=True（防止跨 Episode Reset 突变）
        has_reset = torch.any(self.dones[steps[:, :-1], env_ids.unsqueeze(1)], dim=1)
        # 过滤条件 2：当缓冲区写满循环后，窗口不能跨越当前的写入断点 self.ptr
        if self.size == self.capacity:
            crosses_ptr = (t_starts < self.ptr) & ((t_starts + k) > self.ptr)
        else:
            crosses_ptr = torch.zeros_like(has_reset)

        valid_mask = ~(has_reset | crosses_ptr)
        valid_idx = valid_mask.nonzero(as_tuple=False).squeeze(-1)[:batch_size]
        if valid_idx.numel() < batch_size:
            # 若有效样本略少于 batch_size，允许重复采样补齐
            pad = torch.randint(0, valid_idx.numel(), (batch_size - valid_idx.numel(),), device=self.device)
            valid_idx = torch.cat([valid_idx, valid_idx[pad]], dim=0)

        t_sel = t_starts[valid_idx]
        e_sel = env_ids[valid_idx]
        step_idx = t_sel.unsqueeze(1) + torch.arange(k, device=self.device).unsqueeze(0)  # [B, k]

        # 拼接 k+1 步状态与无噪观测序列
        states_seq = torch.cat(
            [self.states[t_sel, e_sel].unsqueeze(1), self.next_states[step_idx, e_sel.unsqueeze(1)]], dim=1
        )
        clean_obs_seq = torch.cat(
            [self.clean_obs[t_sel, e_sel].unsqueeze(1), self.next_clean_obs[step_idx, e_sel.unsqueeze(1)]], dim=1
        )
        actions_seq = self.actions[step_idx, e_sel.unsqueeze(1)]
        rewards_seq = self.rewards[step_idx, e_sel.unsqueeze(1)]
        dones_seq = self.dones[step_idx, e_sel.unsqueeze(1)]
        commands_seq = self.commands[step_idx, e_sel.unsqueeze(1)]
        contexts_seq = self.contexts[step_idx, e_sel.unsqueeze(1)]

        return states_seq, actions_seq, clean_obs_seq, rewards_seq, dones_seq, commands_seq, contexts_seq

    def sample_initial_states(self, batch_size: int = 4096):
        """采样 H=32 步想象滚动的起始状态 s_0 ~ B (Dyna-style)"""
        t_idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        e_idx = torch.randint(0, self.num_envs, (batch_size,), device=self.device)
        return (
            self.states[t_idx, e_idx].clone(),
            self.obs_history[t_idx, e_idx].clone(),
            self.commands[t_idx, e_idx].clone(),
            self.contexts[t_idx, e_idx].clone(),
        )