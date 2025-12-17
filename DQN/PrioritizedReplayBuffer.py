import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
import random

# SumTree数据结构 - 用于高效实现优先经验回放
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        # 树结构，存储优先级（大小为2*capacity-1）
        self.tree = np.zeros(2 * capacity - 1)
        # 存储实际经验数据
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0
        self.n_entries = 0
        
    def _propagate(self, idx, change):
        """更新树中父节点的值"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx, s):
        """根据优先级和检索样本"""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self):
        """获取所有优先级的总和"""
        return self.tree[0]
    
    def add(self, priority, data):
        """添加新的经验样本"""
        idx = self.data_pointer + self.capacity - 1
        
        self.data[self.data_pointer] = data
        self.update(idx, priority)
        
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def update(self, idx, priority):
        """更新样本的优先级"""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s):
        """根据优先级和获取样本"""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


# 优先经验回放缓冲区
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        """
        Prioritized Replay Buffer
        
        Args:
            capacity: 缓冲区容量
            alpha: 优先级强度，0表示均匀采样，1表示完全优先级
            beta_start: IS权重初始值
            beta_frames: beta达到1所需的帧数
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.beta = beta_start
        self.frame = 1
        self.epsilon = 1e-6  # 防止TD-error为0
        
    def push(self, error, sample):
        """
        添加经验样本
        
        Args:
            error: TD-error
            sample: (state, action, reward, next_state, done)元组
        """
        priority = self._get_priority(error)
        self.tree.add(priority, sample)
        
    def _get_priority(self, error):
        """计算优先级"""
        return (np.abs(error) + self.epsilon) ** self.alpha
        
    def sample(self, batch_size):
        """
        采样一批经验
        
        Args:
            batch_size: 批次大小
        
        Returns:
            indices: 样本索引
            batch: 样本数据
            weights: IS权重
        """
        batch = []
        indices = []
        priorities = []
        
        segment = self.tree.total() / batch_size
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, priority, data = self.tree.get(s)
            indices.append(idx)
            priorities.append(priority)
            batch.append(data)
        
        # 计算IS权重
        sampling_probabilities = np.array(priorities) / self.tree.total()
        self.beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * self.frame / self.beta_frames)
        self.frame += 1
        
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()  # 归一化
        
        # 转换为numpy数组
        states, actions, next_states, rewards, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(next_states),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=np.uint8),
            np.array(indices),
            np.array(is_weights, dtype=np.float32)
        )
    
    def update_priorities(self, indices, errors):
        """
        更新样本优先级
        
        Args:
            indices: 样本索引
            errors: 新的TD-error
        """
        for idx, error in zip(indices, errors):
            priority = self._get_priority(error)
            self.tree.update(idx, priority)
    
    def __len__(self):
        return self.tree.n_entries

