import csv

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

import config
from env import DanmakuVecEnv
from render import Renderer

ACTION_NAMES = ("STOP", "UP", "DOWN", "LEFT", "RIGHT", "UP_LEFT", "UP_RIGHT", "DOWN_LEFT", "DOWN_RIGHT")


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


def save_eval_summary(success_rate, mean_survival_seconds, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"success_rate: {success_rate}\nmean_survival_seconds: {mean_survival_seconds}\n",
        encoding="utf-8",
    )
    return path


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


def save_score_plot(episode_scores, output_path, window=100):
    fig, ax = plt.subplots(figsize=(8, 5))
    episodes = np.arange(1, len(episode_scores) + 1)
    ax.plot(episodes, episode_scores, alpha=0.3, label="score")
    if len(episode_scores) >= window:
        rolling = np.convolve(episode_scores, np.ones(window) / window, mode="valid")
        ax.plot(episodes[window - 1:], rolling, label=f"{window}-episode mean")
    ax.set_title("Episode score")
    ax.set_xlabel("episode")
    ax.set_ylabel("score")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_success_rate_plot(episode_successes, eval_history, output_path, window=100):
    fig, ax = plt.subplots(figsize=(8, 5))
    episodes = np.arange(1, len(episode_successes) + 1)
    if len(episode_successes) >= window:
        rolling = np.convolve(episode_successes, np.ones(window) / window, mode="valid")
        ax.plot(episodes[window - 1:], rolling, label=f"{window}-episode success rate", color="tab:green")
    if eval_history:
        eval_episodes, eval_rates, _eval_survival = zip(*eval_history)
        ax.plot(eval_episodes, eval_rates, marker="o", label="eval success rate", color="tab:orange")
    ax.set_title("Success rate (3-minute survival)")
    ax.set_xlabel("episode")
    ax.set_ylabel("success rate")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_survival_time_plot(episode_survival_seconds, eval_history, output_path, window=100):
    fig, ax = plt.subplots(figsize=(8, 5))
    episodes = np.arange(1, len(episode_survival_seconds) + 1)
    ax.plot(episodes, episode_survival_seconds, alpha=0.3, label="survival seconds", color="tab:red")
    if len(episode_survival_seconds) >= window:
        rolling = np.convolve(episode_survival_seconds, np.ones(window) / window, mode="valid")
        ax.plot(episodes[window - 1:], rolling, label=f"{window}-episode mean", color="darkred")
    if eval_history:
        eval_episodes, _eval_rates, eval_survival = zip(*eval_history)
        ax.plot(eval_episodes, eval_survival, marker="o", label="eval mean survival", color="tab:orange")
    ax.set_title("Survival time")
    ax.set_xlabel("episode")
    ax.set_ylabel("survival seconds")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_loss_plot(episode_losses, output_path, window=100):
    fig, ax = plt.subplots(figsize=(8, 5))
    losses = np.asarray(episode_losses, dtype=np.float64)
    episodes = np.arange(1, len(losses) + 1)
    valid = np.isfinite(losses)
    ax.plot(episodes[valid], losses[valid], alpha=0.3, label="mean_loss (per episode)")
    if valid.sum() >= window:
        rolling = np.convolve(losses[valid], np.ones(window) / window, mode="valid")
        ax.plot(episodes[valid][window - 1:], rolling, label=f"{window}-episode mean")
    ax.set_title("Training loss")
    ax.set_xlabel("episode")
    ax.set_ylabel("loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_action_distribution_plot(action_counts, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = np.asarray(action_counts, dtype=np.float64)
    names = ACTION_NAMES[:len(counts)]
    ax.bar(names, counts, color="tab:purple")
    ax.set_title("Action distribution")
    ax.set_xlabel("action")
    ax.set_ylabel("count")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.3, axis="y")
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
