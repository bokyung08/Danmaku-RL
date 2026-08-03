from __future__ import annotations

import csv
import json
import math
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import config
from env import DanmakuVecEnv
from render import Renderer

ACTION_NAMES = ("STOP", "UP", "DOWN", "LEFT", "RIGHT", "UP_LEFT", "UP_RIGHT", "DOWN_LEFT", "DOWN_RIGHT")


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return path


def open_training_log(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    log_file = path.open("w", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(
        log_file,
        fieldnames=[
            "run", "episode", "global_step", "reward", "learning_reward", "success", "decisions", "physics_steps",
            "survival_seconds", "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "grad_norm",
            "eval_median_survival_seconds",
        ],
    )
    writer.writeheader()
    return log_file, writer


def log_episode(writer, log_file, record):
    writer.writerow(record)
    log_file.flush()


def save_checkpoint(checkpoint, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    return path


def _rolling_mean(values, window):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.asarray([]), np.asarray([])
    effective_window = min(window, values.size)
    means = np.convolve(values, np.ones(effective_window) / effective_window, mode="valid")
    indices = np.arange(effective_window, values.size + 1)
    return indices, means


def save_learning_curve(episode_history, eval_history, update_history, output_path, window=100):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    rewards = [record["reward"] for record in episode_history]
    reward_x, reward_mean = _rolling_mean(rewards, window)
    axes[0, 0].plot(reward_x, reward_mean, label=f"{min(window, max(1, len(rewards)))}-episode mean")
    axes[0, 0].set_title("Episode reward")
    axes[0, 0].set_xlabel("episode")
    axes[0, 0].set_ylabel("raw environment return")

    survival = [record["survival_seconds"] for record in episode_history]
    survival_x, survival_mean = _rolling_mean(survival, window)
    axes[0, 1].plot(survival_x, survival_mean, color="tab:green")
    axes[0, 1].set_title("Survival time")
    axes[0, 1].set_xlabel("episode")
    axes[0, 1].set_ylabel("seconds")

    if eval_history:
        steps = [record["global_step"] for record in eval_history]
        axes[1, 0].plot(steps, [record["median_survival_seconds"] for record in eval_history], marker="o", label="median")
        axes[1, 0].plot(steps, [record["mean_survival_seconds"] for record in eval_history], marker="o", label="mean")
        axes[1, 0].plot(steps, [record["p10_survival_seconds"] for record in eval_history], marker="o", label="p10")
    axes[1, 0].set_title("Deterministic evaluation")
    axes[1, 0].set_xlabel("environment decisions")
    axes[1, 0].set_ylabel("seconds")
    axes[1, 0].legend()

    if update_history:
        steps = [record["global_step"] for record in update_history]
        axes[1, 1].plot(steps, [record["policy_loss"] for record in update_history], label="policy loss")
        axes[1, 1].plot(steps, [record["entropy"] for record in update_history], label="entropy")
        axes[1, 1].plot(steps, [record["clip_fraction"] for record in update_history], label="clip fraction")
    axes[1, 1].set_title("PPO update diagnostics")
    axes[1, 1].set_xlabel("environment decisions")
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.grid(alpha=0.3)
    fig.tight_layout()
    output_path = Path(output_path)
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
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_play_gif(act_fn, output_path, seed, expected_physics_steps, max_frames=1200, size=(240, 240)):
    env = DanmakuVecEnv()
    renderer = Renderer(render_mode="rgb_array")
    observation, _ = env.reset(seed=seed)
    expected_decisions = math.ceil(expected_physics_steps / config.N_FRAME_SKIP)
    capture_stride = max(1, math.ceil(expected_decisions / max_frames))
    frames = []
    captured_physics_steps = []

    def capture_frame():
        rgb = renderer.get_image(env.game, view_score=True)
        frame = Image.fromarray(rgb).resize(size, Image.Resampling.BILINEAR)
        frames.append(frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=128))
        captured_physics_steps.append(int(env.game.state.steps))

    capture_frame()
    decisions = 0
    episode_return = 0.0
    final_info = {"steps": 0, "score": 0}
    final_terminated = False
    final_truncated = False

    while True:
        action = int(act_fn(observation))
        observation, reward, terminated, truncated, info = env.step(action)
        decisions += 1
        episode_return += float(reward)
        if decisions % capture_stride == 0 or terminated or truncated:
            capture_frame()
        if terminated or truncated:
            final_info = info
            final_terminated = terminated
            final_truncated = truncated
            break

    renderer.close()
    if int(final_info["steps"]) != int(expected_physics_steps):
        raise RuntimeError(f"GIF replay ended at {final_info['steps']} physics steps; expected {expected_physics_steps}")

    frame_durations_ms = []
    exact_elapsed_ms = 0.0
    stored_elapsed_ms = 0
    for current_step, next_step in pairwise(captured_physics_steps):
        exact_elapsed_ms += 1000 * (next_step - current_step) / config.PHYSICS_FPS
        target_stored_ms = round(exact_elapsed_ms / 10) * 10
        duration_ms = max(10, target_stored_ms - stored_elapsed_ms)
        frame_durations_ms.append(duration_ms)
        stored_elapsed_ms += duration_ms

    terminal_hold_ms = 10
    for index in range(len(frame_durations_ms) - 1, -1, -1):
        if frame_durations_ms[index] > terminal_hold_ms:
            frame_durations_ms[index] -= terminal_hold_ms
            break
    else:
        raise RuntimeError("GIF timing cannot reserve the terminal frame")
    frame_durations_ms.append(terminal_hold_ms)

    expected_duration_ms = round(1000 * int(final_info["steps"]) / config.PHYSICS_FPS / 10) * 10
    if sum(frame_durations_ms) != expected_duration_ms:
        raise RuntimeError("GIF duration does not match the evaluated episode")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_durations_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )
    return {
        "path": str(output_path),
        "seed": int(seed),
        "frames": len(frames),
        "capture_stride": capture_stride,
        "expected_duration_ms": expected_duration_ms,
        "gif_total_duration_ms": sum(frame_durations_ms),
        "timing_error_ms": 0,
        "decisions": decisions,
        "physics_steps": int(final_info["steps"]),
        "survival_seconds": float(final_info["steps"] / config.PHYSICS_FPS),
        "score": int(final_info["score"]),
        "return": episode_return,
        "terminated": bool(final_terminated),
        "truncated": bool(final_truncated),
    }


def save_policy_analysis(analyze_fn, image_path, csv_path, seed, max_steps=None):
    env = DanmakuVecEnv()
    observation, _ = env.reset(seed=seed)
    max_steps = max_steps or config.MAX_TIME_STEPS // config.N_FRAME_SKIP + 1
    records = []

    for decision in range(max_steps):
        action, probabilities, value = analyze_fn(observation)
        entropy = float(-(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))).sum())
        records.append(
            {
                "decision": decision,
                "physics_steps": int(env.game.state.steps),
                "seconds": float(env.game.state.steps / config.PHYSICS_FPS),
                "selected_action": int(action),
                "selected_action_name": ACTION_NAMES[int(action)],
                "state_value": float(value),
                "entropy": entropy,
                "probabilities": np.asarray(probabilities, dtype=np.float64),
            }
        )
        observation, _reward, terminated, truncated, _info = env.step(int(action))
        if terminated or truncated:
            break

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        fieldnames = [
            "decision", "physics_steps", "seconds", "selected_action", "selected_action_name", "state_value", "entropy",
            *[f"prob_{name}" for name in ACTION_NAMES],
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {key: value for key, value in record.items() if key != "probabilities"}
            row.update({f"prob_{name}": float(record["probabilities"][index]) for index, name in enumerate(ACTION_NAMES)})
            writer.writerow(row)

    probabilities = np.stack([record["probabilities"] for record in records], axis=1)
    times = np.asarray([record["seconds"] for record in records], dtype=np.float64)
    values = np.asarray([record["state_value"] for record in records], dtype=np.float64)
    actions = np.asarray([record["selected_action"] for record in records], dtype=np.int64)
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    image = axes[0].imshow(probabilities, aspect="auto", origin="upper", vmin=0.0, vmax=1.0, cmap="viridis")
    axes[0].scatter(np.arange(actions.size), actions, s=8, c="black", label="selected action")
    axes[0].set_yticks(range(len(ACTION_NAMES)), ACTION_NAMES)
    axes[0].set_ylabel("action")
    axes[0].set_title(f"PPO action probabilities — seed {seed}")
    axes[0].legend(loc="upper right")
    fig.colorbar(image, ax=axes[0], label="probability")
    axes[1].plot(values, color="tab:blue", label="state value")
    axes[1].set_ylabel("V(s)")
    axes[1].set_xlabel("greedy trajectory time")
    tick_indices = np.linspace(0, max(0, len(records) - 1), min(8, len(records)), dtype=int)
    axes[1].set_xticks(tick_indices, [f"{times[index]:.1f}s" for index in tick_indices])
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    image_path = Path(image_path)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(image_path, dpi=150)
    plt.close(fig)
    return image_path, csv_path, records
