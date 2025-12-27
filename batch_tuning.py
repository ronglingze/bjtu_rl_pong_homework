import os
from DQN.dqn import DQNAgent
from tuning_module import hyperparam_tuning  # 调参函数
import pickle
import matplotlib.pyplot as plt

# 定义多组超参数组合和每组对应的训练 episode 数
hyperparams_list = [
    #{"params": {"lr": 1e-4, "batch_size": 64, "std_init": 0.4}, "total_episode": 100},
    #{"params": {"lr": 5e-5, "batch_size": 64, "std_init": 0.4}, "total_episode": 100},
    #{"params": {"lr": 2e-4, "batch_size": 64, "std_init": 0.4}, "total_episode": 100},
    #{"params": {"lr": 1e-4, "batch_size": 32, "std_init": 0.4}, "total_episode": 100},
    #{"params": {"lr": 5e-5, "batch_size": 32, "std_init": 0.4}, "total_episode": 100},
    #{"params": {"lr": 2e-4, "batch_size": 32, "std_init": 0.4}, "total_episode": 100},
    {"params": {"lr": 5e-5, "batch_size": 64, "std_init": 0.1, "gamma": 0.99}, "total_episode": 100},
    {"params": {"lr": 5e-5, "batch_size": 64, "std_init": 0.2, "gamma": 0.99}, "total_episode": 100},
    {"params": {"lr": 5e-5, "batch_size": 64, "std_init": 0.1, "gamma": 0.995}, "total_episode": 100},
    {"params": {"lr": 5e-5, "batch_size": 64, "std_init": 0.2, "gamma": 0.995}, "total_episode": 100},
]

# 保存不同组合日志和模型的目录
base_log_dir = 'logs_batch_tuning'
os.makedirs(base_log_dir, exist_ok=True)
results = []

# 循环训练每组参数
for idx, param_entry in enumerate(hyperparams_list):
    param_dict = param_entry["params"]
    total_episode = param_entry["total_episode"]

    print(f"\n=== Training hyperparameter set {idx+7}/{len(hyperparams_list)} ===")
    print(f"Parameters: {param_dict}, Total Episode: {total_episode}")

    log_dir = os.path.join(base_log_dir, f"set_{idx+7}")
    os.makedirs(log_dir, exist_ok=True)

    # 调用调参函数
    agent, log_file = hyperparam_tuning(
        agent_class=DQNAgent,
        param_dict=param_dict,
        player=1,
        total_episode=total_episode,
        log_dir=log_dir
    )

    results.append({
        "params": param_dict,
        "total_episode": total_episode,
        "agent": agent,
        "log_file": log_file
    })

# 绘制两个子图：Episode Loss / Episode Q
fig, axes = plt.subplots(1, 2, figsize=(12,5))
ax_ep_loss, ax_ep_q = axes

for idx, res in enumerate(results):
    with open(res["log_file"], "rb") as f:
        data = pickle.load(f)

    # Episode 级别
    ax_ep_loss.plot(data["losses_ep"], label=f"Loss set_{idx+7}")
    ax_ep_q.plot(data["q_means_ep"], label=f"Q set_{idx+7}")

# 设置标签和标题
ax_ep_loss.set_xlabel("Episode")
ax_ep_loss.set_ylabel("Avg TD Loss")
ax_ep_loss.set_title("Episode Loss Comparison")
ax_ep_loss.legend()

ax_ep_q.set_xlabel("Episode")
ax_ep_q.set_ylabel("Avg Q Value")
ax_ep_q.set_title("Episode Q Value Comparison")
ax_ep_q.legend()

plt.tight_layout()

# 保存图像，不展示
save_path = os.path.join(base_log_dir, "hyperparam_comparison_episode_gamma&std.png")
plt.savefig(save_path)
plt.close(fig)
print(f"[INFO] Hyperparameter episode-level comparison figure saved to {save_path}")
