import collections
import random
import numpy as np

# 定义一个存储样本的命名元组
Experience = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward', 'next_state', 'done']
)

class ReplayBuffer:
    """
    基础经验回放缓冲区
    """
    def __init__(self, capacity):
        """
        初始化 Replay Buffer
        :param capacity: 缓冲区的最大容量
        """
        self.buffer = collections.deque(maxlen=capacity)
        self.capacity = capacity

    def __len__(self):
        """ 返回当前缓冲区中存储的样本数量 """
        return len(self.buffer)

    def append(self, experience):
        """
        向缓冲区中添加一个经验样本
        :param experience: Experience 命名元组 (state, action, reward, next_state, done)
        """
        self.buffer.append(experience)

    def sample(self, batch_size):
        """
        从缓冲区中随机采样一个批次 (batch_size) 的样本
        :param batch_size: 采样的批次大小
        :return: 包含 batch_size 个 Experience 的列表
        """
        if len(self.buffer) < batch_size:
            # 当缓冲区样本不足时，返回所有样本
            return list(self.buffer)
        
        # 使用 random.sample 进行不重复的随机采样
        samples = random.sample(self.buffer, batch_size)
        return samples

    def save(self, file_path):
        """
        数据持久化: 将缓冲区中的数据保存到文件
        :param file_path: 保存文件的路径 (推荐使用 .pkl)
        """
        # (待实现) C 成员需要选择合适的序列化方式 (如 pickle 或 HDF5) 来保存 self.buffer
        print(f"✅ Replay Buffer 已保存至: {file_path} (待实现具体存储逻辑)")

    def load(self, file_path):
        """
        数据持久化: 从文件中加载数据到缓冲区
        :param file_path: 加载文件的路径
        """
        # (待实现) C 成员需要实现从文件反序列化数据到 self.buffer 的逻辑
        print(f"✅ Replay Buffer 已从 {file_path} 加载 (待实现具体加载逻辑)")


class PrioritizedReplayBuffer:
    """
    优先经验回放缓冲区 (Prioritized Experience Replay, PER)
    使用 Proportional prioritization, 采样概率 P(i) ~ |delta_i|^alpha
    """
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001, epsilon=1e-6):
        """
        :param capacity: 缓冲区容量
        :param alpha: 决定 TD-Error 优先级程度的参数 (0=均匀采样, 1=完全基于TD-Error)
        :param beta: 用于重要性采样 (IS) 权重的参数，随训练增加，以纠正非均匀采样带来的偏差
        :param beta_increment_per_sampling: 每次采样后 beta 的增量
        :param epsilon: 防止 TD-Error 为 0 时优先级为 0 的小量
        """
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)
        # 存储每个样本的优先级，长度与 buffer 相同
        self.priorities = collections.deque(maxlen=capacity) 
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.epsilon = epsilon
        # 初始时，所有样本使用最大优先级，保证初始采样概率
        self.max_priority = 1.0 

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        """
        添加新样本时，赋予其当前已有的最大优先级
        """
        self.buffer.append(experience)
        self.priorities.append(self.max_priority) # 新样本赋予最高优先级

    def sample(self, batch_size):
        """
        基于优先级采样样本并计算重要性采样权重
        :return: 样本列表, IS权重数组, 样本索引数组
        """
        # 1. 计算每个样本的采样概率 P(i)
        # P(i) = (priority_i^alpha / sum(priority_j^alpha))
        priorities_np = np.array(self.priorities, dtype=np.float32)
        
        # 优先级的 alpha 次方
        prob_alpha = priorities_np ** self.alpha
        
        # 采样概率 P(i)
        probabilities = prob_alpha / np.sum(prob_alpha)
        
        # 2. 基于概率进行采样 (替换了 random.sample)
        # p=probabilities 确保采样是基于优先级的
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # 提取样本
        samples = [self.buffer[idx] for idx in indices]
        
        # 3. 计算重要性采样 (IS) 权重 w_i
        # w_i = (1/N * 1/P(i))^beta / max(w_j)
        
        # 找到最小概率 P_min = 1 / capacity^beta * max(w_j) 的分母
        # (1/capacity) 是均匀采样时的概率
        min_prob = np.min(probabilities)
        max_weight = (min_prob * len(self.buffer)) ** (-self.beta)
        
        # IS 权重
        weights = ((len(self.buffer) * probabilities[indices]) ** (-self.beta)) / max_weight
        
        # 4. 更新 beta 值（重要性，随训练增加）
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)
        
        return samples, weights, indices

    def update_priorities(self, indices, td_errors):
        """
        在 Q 网络训练后，使用新的 TD-Error 更新样本的优先级
        :param indices: 采样的样本索引
        :param td_errors: 对应样本的新 TD-Error (必须是正值，所以通常取绝对值)
        """
        new_priorities = np.abs(td_errors) + self.epsilon
        
        # 更新 self.priorities
        for idx, new_p in zip(indices, new_priorities):
            self.priorities[idx] = new_p
            
        # 更新 max_priority
        self.max_priority = max(self.max_priority, np.max(new_priorities))