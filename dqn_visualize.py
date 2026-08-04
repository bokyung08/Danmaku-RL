import csv

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

import config
from env import DanmakuVecEnv
from render import Renderer


def open_training_log(path):
    log_file = path.open("w", newline="", encoding="utf-8-sig")
    writer = csv.writer(log_file)
    writer.writerow(
        ["run", "episode", "reward", "success", "steps", "mean_loss", "eps", "eval_success_rate", "eval_mean_survival_seconds"]
    )
    return log_file, writer


def log_episode(writer, log_file, run_id, episode, reward, success, steps, mean_loss, eps, eval_success_rate, eval_mean_survival_seconds):
    writer.writerow([run_id, episode, reward, success, steps, mean_loss, eps, eval_success_rate, eval_mean_survival_seconds])
    log_file.flush()  # 중간에 학습이 중단되어도 여기까지는 파일에 남도록 즉시 기록


def save_best_model(state_dict, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, path)


def save_learning_curve(episode_rewards, episode_successes, eval_history, output_path, window=100):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    episodes = np.arange(1, len(episode_rewards) + 1)
    axes[0].plot(episodes, episode_rewards, alpha=0.3, label="reward")
    if len(episode_rewards) >= window:
        rolling_reward = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
        axes[0].plot(episodes[window - 1:], rolling_reward, label=f"{window}-episode mean")
    axes[0].set_ylabel("episode reward")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    if len(episode_successes) >= window:
        rolling_success = np.convolve(episode_successes, np.ones(window) / window, mode="valid")
        axes[1].plot(episodes[window - 1:], rolling_success, label=f"{window}-episode success rate", color="tab:green")
    axes[1].set_ylabel("success rate")
    axes[1].set_xlabel("episode")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(alpha=0.3)

    if eval_history:
        eval_episodes, eval_rates, eval_survival = zip(*eval_history)
        axes[1].plot(eval_episodes, eval_rates, marker="o", label="eval success rate", color="tab:orange")
        survival_axis = axes[1].twinx()
        survival_axis.plot(eval_episodes, eval_survival, marker="o", label="eval mean survival seconds", color="tab:red")
        survival_axis.set_ylabel("survival seconds")
        lines, labels = axes[1].get_legend_handles_labels()
        survival_lines, survival_labels = survival_axis.get_legend_handles_labels()
        axes[1].legend(lines + survival_lines, labels + survival_labels, loc="upper left", fontsize=8)
    else:
        axes[1].legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_reward_curve(all_runs_episode_rewards, output_path, window=100, title="Danmaku-RL"):
    fig, ax = plt.subplots(figsize=(8, 6))

    min_length = min(len(run) for run in all_runs_episode_rewards)
    runs = np.asarray([run[:min_length] for run in all_runs_episode_rewards], dtype=np.float64)
    effective_window = min(window, min_length)

    smoothed = np.stack(
        [np.convolve(run, np.ones(effective_window) / effective_window, mode="valid") for run in runs]
    )
    episodes = np.arange(effective_window, min_length + 1)

    mean = smoothed.mean(axis=0)
    std = smoothed.std(axis=0)

    ax.plot(episodes, mean, color="tab:blue")
    ax.fill_between(episodes, mean - std, mean + std, alpha=0.25, color="tab:blue")
    ax.set_title(title)
    ax.set_xlabel("episode")
    ax.set_ylabel(f"reward ({effective_window}-episode mean)")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_play_video(act_fn, output_path, seed=0, max_frames=None, size=(240, 240)):
    env = DanmakuVecEnv()
    renderer = Renderer(render_mode="rgb_array")
    observation, _ = env.reset(seed=seed)

    if max_frames is None:
        max_frames = config.MAX_TIME_STEPS // config.N_FRAME_SKIP + 1
    fps = config.PHYSICS_FPS / config.N_FRAME_SKIP  # env.step() 1회 = 물리 프레임 N_FRAME_SKIP개 분량의 실제 시간

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    def capture():
        rgb = renderer.get_image(env.game, view_score=True)
        resized = cv2.resize(rgb, size, interpolation=cv2.INTER_LINEAR)
        writer.write(cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))

    capture()
    frame_count = 1
    while frame_count < max_frames:
        action = act_fn(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        capture()
        frame_count += 1
        if terminated or truncated:
            break

    renderer.close()
    writer.release()
    return output_path
