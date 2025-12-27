import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import retro
from DQN.dqn import DQNAgent
from multiplayer import PongDiscretizer  # 环境相关工具
from config import *

CONFIG = {
    'model_dir': 'checkpoints',
    'video_dir': 'videos',
}

def hyperparam_tuning(agent_class, param_dict, 
                      player=1, skip_frame=4, horizon=4, 
                      max_steps=2500, start_episode=0, total_episode=50,
                      log_dir='tuning_logs'):
    """
    调参专用函数：训练、保存 step/episode 曲线和日志
    """

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(CONFIG['model_dir'], exist_ok=True)

    # 初始化 agent
    agent = agent_class(
        state_size=(horizon, 84, 84),
        action_size=3,
        skip_frame=skip_frame,
        horizon=horizon,
        **param_dict
    )

    # 日志列表
    episode_rewards = []
    losses_ep = []      # episode 平均 loss
    q_means_ep = []     # episode 平均 q
    losses_step = []    # step 级 loss
    q_means_step = []   # step 级 q
    steps_list = []     # 每个 episode 的 step 数

    env = PongDiscretizer(retro.make(game='Pong-Atari2600', players=player), players=player)
    env.reset()

    pre_action = 2
    global_step = 0

    for ep in range(start_episode, start_episode + total_episode):
        obs = env.reset()
        total_rew = 0.0
        skip_rew = 0.0
        done = False

        agent.reset()

        ep_losses = []
        ep_qs = []
        ep_steps = 0

        while not done:
            eps = agent.update_epsilon(global_step)
            action = agent.select_action(obs, eps)
            nxt_obs, rew, done, info = env.step(action)
            total_rew += rew
            skip_rew += rew

            if agent.dqn_net.obs_process_tool.frame_cnt == 0:
                agent.memory_push(agent.dqn_net.obs_process_tool.pre_obs, pre_action,
                                  agent.dqn_net.obs_process_tool.obs, skip_rew, done)
                loss_val, q_mean_val = agent.update(global_step, return_values=True)

                # 转标量
                loss_val = float(np.nan_to_num(np.mean(loss_val)))
                q_mean_val = float(np.nan_to_num(np.mean(q_mean_val)))


                ep_losses.append(loss_val)
                ep_qs.append(q_mean_val)
                losses_step.append(loss_val)
                q_means_step.append(q_mean_val)

                pre_action = action
                global_step += 1
                ep_steps += 1
                skip_rew = 0.0

            obs = nxt_obs

        # episode 结束后记录
        episode_rewards.append(total_rew)
        losses_ep.append(np.mean(ep_losses) if ep_losses else 0.0)
        q_means_ep.append(np.mean(ep_qs) if ep_qs else 0.0)
        steps_list.append(ep_steps)

        print(f"[EP {ep}] Total Reward: {total_rew:.2f}, Steps: {ep_steps}, Epsilon: {eps:.4f}")

        #print(f"step_losses = {ep_losses}")
        #print(f"step_qs = {ep_qs}")

        # 保存 step 级曲线（不展示）
        fig_step, (ax_loss, ax_q) = plt.subplots(1, 2, figsize=(12,4))
        ax_loss.plot(ep_losses)
        ax_loss.set_title(f"TD Loss per Step (Episode {ep})")
        ax_loss.set_xlabel("Step")
        ax_loss.set_ylabel("Loss")
        ax_q.plot(ep_qs)
        ax_q.set_title(f"Q Value per Step (Episode {ep})")
        ax_q.set_xlabel("Step")
        ax_q.set_ylabel("Q Mean")
        plt.tight_layout()
        step_save_path = os.path.join(log_dir, f"step_curve_ep{ep}.png")
        fig_step.savefig(step_save_path)
        plt.close(fig_step)

    # 保存训练日志
    log_file = os.path.join(log_dir, f"log_{start_episode}_{start_episode + total_episode}.pkl")
    with open(log_file, "wb") as f:
        pickle.dump({
            "steps": steps_list,
            "reward": episode_rewards,
            "losses_ep": losses_ep,
            "q_means_ep": q_means_ep,
            "losses_step": losses_step,
            "q_means_step": q_means_step,
            "params": param_dict
        }, f)
    print(f"[INFO] Training log saved to {log_file}")

    # 保存模型
    agent.save_model(start_episode + total_episode, CONFIG['model_dir'])
    print(f"[INFO] Model saved to {CONFIG['model_dir']}")

    #print(f"losses_ep = {losses_ep}")
    #print(f"q_means_ep = {q_means_ep}")

    # 保存 episode 级曲线（不展示），三个子图：Reward / Avg TD Loss / Avg Q Value
    fig_ep, axes = plt.subplots(1, 3, figsize=(18,5))
    ax_reward, ax_loss_ep, ax_q_ep = axes
    # Total Reward per Episode
    ax_reward.plot(episode_rewards)
    ax_reward.set_title("Total Reward per Episode")
    ax_reward.set_xlabel("Episode")
    ax_reward.set_ylabel("Reward")
    # Avg TD Loss per Episode
    ax_loss_ep.plot(losses_ep, label="Avg TD Loss", color="orange")
    ax_loss_ep.set_title("Average TD Loss per Episode")
    ax_loss_ep.set_xlabel("Episode")
    ax_loss_ep.set_ylabel("Loss")
    ax_loss_ep.legend()
    # Avg Q Value per Episode
    ax_q_ep.plot(q_means_ep, label="Avg Q Value", color="green")
    ax_q_ep.set_title("Average Q Value per Episode")
    ax_q_ep.set_xlabel("Episode")
    ax_q_ep.set_ylabel("Q Mean")
    ax_q_ep.legend()

    plt.tight_layout()
    ep_save_path = os.path.join(log_dir, f"episode_curve_{start_episode}_{start_episode + total_episode}.png")
    fig_ep.savefig(ep_save_path)
    plt.close(fig_ep)
    print(f"[INFO] Episode-level training curve saved to {ep_save_path}")


    return agent, log_file

