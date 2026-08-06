import csv
import json
import math
import random
import shutil
import sys
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from env import DanmakuVecEnv
from render import Renderer

ACTION_NAMES = ("STOP", "UP", "DOWN", "LEFT", "RIGHT", "UP_LEFT", "UP_RIGHT", "DOWN_LEFT", "DOWN_RIGHT")


# ------------------------------------------------------------
# 1. 시드 / 환경 검증 / 기록 저장 함수
# ------------------------------------------------------------

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_vec_env(env, seed):
    observation, info = env.reset(seed=seed)

    if not isinstance(observation, np.ndarray):
        raise TypeError("reset observation must be a NumPy array")
    if observation.shape != env.observation_shape:
        raise ValueError(f"observation shape {observation.shape} != {env.observation_shape}")
    if observation.ndim != 1:
        raise ValueError(f"vector observation must be 1-D, got {observation.ndim}")
    if observation.dtype != np.float32:
        raise TypeError(f"observation dtype must be float32, got {observation.dtype}")
    if not np.isfinite(observation).all():
        raise ValueError("observation contains NaN or infinity")
    if observation.min() < -1.00001 or observation.max() > 1.00001:
        raise ValueError(f"observation is outside [-1, 1]: min={observation.min()}, max={observation.max()}")

    actions = tuple(env.action_space)
    if actions != tuple(range(len(actions))):
        raise ValueError(f"action space must be consecutive from 0, got {actions}")
    if not isinstance(info, dict):
        raise TypeError("reset info must be a dictionary")

    initial_physics_steps = int(info["steps"])
    _, _, terminated, truncated, probe_info = env.step(actions[0])
    physics_steps_per_decision = int(probe_info["steps"]) - initial_physics_steps
    if terminated or truncated:
        raise RuntimeError("environment ended during the initial frame-skip probe")
    if physics_steps_per_decision <= 0:
        raise ValueError("one environment decision must advance at least one physics step")

    return {
        "obs_dim": int(observation.size),
        "n_actions": len(actions),
        "dtype": str(observation.dtype),
        "observation_shape": tuple(observation.shape),
        "physics_steps_per_decision": physics_steps_per_decision,
        "configured_frame_skip": int(config.N_FRAME_SKIP),
        "physics_fps": int(config.PHYSICS_FPS),
        "max_time_steps": int(config.MAX_TIME_STEPS),
    }


def make_run_dir(base_dir, algorithm, seed):
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / algorithm / f"seed_{seed}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(data), ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(_jsonable(record), ensure_ascii=False))
        file.write("\n")


# ------------------------------------------------------------
# 2. 체크포인트 / 평가 로그 저장 함수
# ------------------------------------------------------------

def save_checkpoint(path, seed, global_step, total_physics_steps, episode_index, env_spec, evaluation_summary,
                     network_type, hyperparameters, online_net, target_net, optimizer, action_rng):
    checkpoint = {
        "algorithm": "Dueling Double DQN" if network_type == "dueling" else "Double DQN",
        "seed": seed,
        "global_step": global_step,
        "total_physics_steps": total_physics_steps,
        "episode_index": episode_index,
        "hyperparameters": hyperparameters,
        "env_spec": env_spec,
        "action_names": ACTION_NAMES,
        "online_state_dict": online_net.state_dict(),
        "target_state_dict": target_net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "evaluation_summary": evaluation_summary,
        "python_random_state": random.getstate(),
        "numpy_action_rng_state": action_rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        checkpoint["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def log_evaluation(run_evaluation_fn, eval_episodes, eval_seed_start, seed, run_dir, metrics_path, episode_index,
                    global_step, total_physics_steps, best_checkpoint, best_score, best_summary, env_spec,
                    network_type, hyperparameters, online_net, target_net, optimizer, action_rng):
    records, summary = run_evaluation_fn(eval_episodes, eval_seed_start)
    evaluation_record = {
        "type": "evaluation",
        "train_seed": seed,
        "episode_index": episode_index,
        "global_step": global_step,
        "summary": summary,
        "episodes": records,
    }
    append_jsonl(metrics_path, evaluation_record)
    save_json(run_dir / "latest_evaluation.json", evaluation_record)

    score = (summary["median_survival_seconds"], summary["p10_survival_seconds"], summary["mean_survival_seconds"])
    if score > best_score:
        best_score = score
        best_summary = summary
        save_checkpoint(best_checkpoint, seed, global_step, total_physics_steps, episode_index, env_spec, summary,
                         network_type, hyperparameters, online_net, target_net, optimizer, action_rng)
        save_json(run_dir / "best_evaluation.json", evaluation_record)

    return evaluation_record, best_score, best_summary


# ------------------------------------------------------------
# 3. 시각화 / 아티팩트 저장 함수
# ------------------------------------------------------------

def rolling_mean(values, window=50):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return np.asarray([]), np.asarray([])
    effective_window = min(window, array.size)
    kernel = np.ones(effective_window, dtype=np.float64) / effective_window
    means = np.convolve(array, kernel, mode="valid")
    indices = np.arange(effective_window - 1, array.size)
    return indices, means


def save_learning_curves(results, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    eval_medians = []
    eval_means = []
    shared_eval_episodes = None

    for result in results:
        seed = result["seed"]
        evaluations = result["evaluation_history"]
        eval_episodes_axis = np.asarray([record["episode_index"] for record in evaluations], dtype=np.int64)
        medians = np.asarray([record["summary"]["median_survival_seconds"] for record in evaluations], dtype=np.float64)
        means = np.asarray([record["summary"]["mean_survival_seconds"] for record in evaluations], dtype=np.float64)
        axes[0, 0].plot(eval_episodes_axis, medians, marker="o", label=f"seed {seed}")
        axes[0, 1].plot(eval_episodes_axis, means, marker="o", label=f"seed {seed}")
        eval_medians.append(medians)
        eval_means.append(means)
        if shared_eval_episodes is None:
            shared_eval_episodes = eval_episodes_axis

        episodes = result["episode_history"]
        episode_numbers = [record["episode_index"] + 1 for record in episodes]
        survival = [record["survival_seconds"] for record in episodes]
        rolling_indices, rolling_values = rolling_mean(survival, window=500)
        if rolling_values.size:
            x_values = np.asarray(episode_numbers)[rolling_indices]
            axes[1, 0].plot(x_values, rolling_values, alpha=0.8, label=f"seed {seed}")

        returns = [record["training_return"] for record in episodes]
        return_indices, return_values = rolling_mean(returns, window=500)
        if return_values.size:
            x_values = np.asarray(episode_numbers)[return_indices]
            axes[1, 1].plot(x_values, return_values, alpha=0.8, label=f"seed {seed}")

    if (
        len(results) > 1
        and shared_eval_episodes is not None
        and eval_medians
        and len({len(item) for item in eval_medians}) == 1
    ):
        axes[0, 0].plot(
            shared_eval_episodes, np.mean(np.stack(eval_medians), axis=0),
            color="black", linewidth=3, linestyle="--", label="seed mean",
        )
        axes[0, 1].plot(
            shared_eval_episodes, np.mean(np.stack(eval_means), axis=0),
            color="black", linewidth=3, linestyle="--", label="seed mean",
        )

    titles = (
        "Validation median survival", "Validation mean survival",
        "Training survival (500-episode rolling mean)", "Shaped training return (500-episode rolling mean)",
    )
    y_labels = ("seconds", "seconds", "seconds", "return")
    for axis, title, y_label in zip(axes.flat, titles, y_labels):
        axis.set_title(title)
        axis.set_xlabel("completed training episodes")
        axis.set_ylabel(y_label)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)

    seed_text = ", ".join(str(result["seed"]) for result in results)
    fig.suptitle(f"Dueling Double DQN learning curves — train seed {seed_text}", fontsize=16)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def collect_q_trajectory(q_values_fn, seed):
    traj_env = DanmakuVecEnv()
    state, _ = traj_env.reset(seed=seed)
    q_value_list, action_list, physics_step_list = [], [], []
    episode_return = 0.0

    while True:
        q_row = q_values_fn(state)
        action = int(np.argmax(q_row))
        q_value_list.append(q_row)
        action_list.append(action)
        physics_step_list.append(int(traj_env.game.state.steps))

        state, reward, terminated, truncated, info = traj_env.step(action)
        episode_return += float(reward)
        if terminated or truncated:
            return {
                "seed": int(seed),
                "q_values": np.stack(q_value_list),
                "actions": np.asarray(action_list, dtype=np.int64),
                "physics_steps": np.asarray(physics_step_list, dtype=np.int64),
                "return": episode_return,
                "score": int(info["score"]),
                "final_physics_steps": int(info["steps"]),
                "survival_seconds": float(info["steps"] / config.PHYSICS_FPS),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }


def save_q_table_visualization(trajectory, image_path, csv_path, max_rows=1_200):
    traj_q_values = trajectory["q_values"]
    actions = trajectory["actions"]
    physics_steps = trajectory["physics_steps"]

    if len(traj_q_values) > max_rows:
        indices = np.linspace(0, len(traj_q_values) - 1, max_rows, dtype=np.int64)
    else:
        indices = np.arange(len(traj_q_values))

    sampled_q = traj_q_values[indices]
    sampled_actions = actions[indices]
    sampled_seconds = physics_steps[indices] / config.PHYSICS_FPS
    centered_q = sampled_q - sampled_q.mean(axis=1, keepdims=True)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    raw_image = axes[0].imshow(sampled_q.T, aspect="auto", interpolation="nearest", cmap="viridis")
    axes[0].set_title("Q-network outputs along greedy trajectory")
    axes[0].set_yticks(range(len(ACTION_NAMES)))
    axes[0].set_yticklabels(ACTION_NAMES)
    axes[0].set_ylabel("action")
    fig.colorbar(raw_image, ax=axes[0], label="Q(s, a)")

    centered_image = axes[1].imshow(centered_q.T, aspect="auto", interpolation="nearest", cmap="coolwarm")
    axes[1].scatter(np.arange(len(sampled_actions)), sampled_actions, s=4, c="black", label="greedy action")
    axes[1].set_title("Per-state centered Q values and selected action")
    axes[1].set_yticks(range(len(ACTION_NAMES)))
    axes[1].set_yticklabels(ACTION_NAMES)
    axes[1].set_ylabel("action")
    axes[1].set_xlabel("greedy trajectory time")
    axes[1].legend(loc="upper right")
    fig.colorbar(centered_image, ax=axes[1], label="Q(s, a) − state mean")

    tick_count = min(8, len(sampled_seconds))
    if tick_count:
        tick_positions = np.linspace(0, len(sampled_seconds) - 1, tick_count, dtype=np.int64)
        axes[1].set_xticks(tick_positions)
        axes[1].set_xticklabels([f"{sampled_seconds[index]:.1f}s" for index in tick_positions])

    fig.suptitle(
        "Neural Q-value heatmap (continuous state; not a tabular Q-table)\n"
        f"seed={trajectory['seed']}, survival={trajectory['survival_seconds']:.2f}s, score={trajectory['score']}",
        fontsize=14,
    )
    fig.tight_layout()
    image_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(image_path, dpi=160)
    plt.close(fig)

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["decision", "physics_steps", "seconds", "action", *ACTION_NAMES])
        for decision, (step, action, q_row) in enumerate(zip(physics_steps, actions, traj_q_values)):
            writer.writerow(
                [decision, int(step), float(step / config.PHYSICS_FPS), int(action), *[float(value) for value in q_row]]
            )

    return image_path, csv_path


def save_play_gif(deterministic_action_fn, seed, output_path, expected_physics_steps, physics_steps_per_decision,
                   max_frames=1_200, size=(240, 240)):
    play_env = DanmakuVecEnv()
    renderer = Renderer(render_mode="rgb_array")
    state, _ = play_env.reset(seed=seed)

    expected_decisions = math.ceil(expected_physics_steps / physics_steps_per_decision)
    capture_stride = max(1, math.ceil(expected_decisions / max_frames))
    frames = []
    captured_physics_steps = []

    def capture_frame():
        rgb = renderer.get_image(play_env.game, view_score=True)
        frame = Image.fromarray(rgb).resize(size, Image.Resampling.BILINEAR)
        frames.append(frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=128))
        captured_physics_steps.append(int(play_env.game.state.steps))

    capture_frame()
    decision = 0
    episode_return = 0.0
    final_info = {"steps": 0, "score": 0}
    final_terminated = False
    final_truncated = False

    while True:
        action = deterministic_action_fn(state)
        state, reward, terminated, truncated, info = play_env.step(action)
        decision += 1
        episode_return += float(reward)

        if decision % capture_stride == 0 or terminated or truncated:
            capture_frame()

        if terminated or truncated:
            final_info = info
            final_terminated = terminated
            final_truncated = truncated
            break

    renderer.close()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if int(final_info["steps"]) != expected_physics_steps:
        raise RuntimeError(
            f"Video replay ended at {final_info['steps']} physics steps, but the evaluated trajectory ended at "
            f"{expected_physics_steps}."
        )

    # GIF는 duration을 10ms 단위로 저장한다. 그냥 반올림하면 66.67ms 간격이 60ms로
    # 굳어져 영상이 빨라지므로, 누적 오차를 프레임마다 재분배한다.
    frame_durations_ms = []
    exact_elapsed_ms = 0.0
    stored_elapsed_ms = 0
    for current_step, next_step in pairwise(captured_physics_steps):
        physics_delta = next_step - current_step
        exact_elapsed_ms += 1000 * physics_delta / config.PHYSICS_FPS
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
        raise RuntimeError("GIF timing cannot reserve a terminal-frame hold without extending the episode duration.")
    frame_durations_ms.append(terminal_hold_ms)

    expected_duration_ms = round(1000 * int(final_info["steps"]) / config.PHYSICS_FPS / 10) * 10
    gif_total_duration_ms = sum(frame_durations_ms)
    if gif_total_duration_ms != expected_duration_ms:
        raise RuntimeError(f"GIF duration is {gif_total_duration_ms} ms; expected {expected_duration_ms} ms.")

    frames[0].save(
        output_path, save_all=True, append_images=frames[1:], duration=frame_durations_ms,
        loop=0, optimize=False, disposal=2,
    )

    return {
        "path": str(output_path),
        "seed": int(seed),
        "frames": len(frames),
        "capture_stride": capture_stride,
        "physics_steps_per_decision": physics_steps_per_decision,
        "nominal_capture_duration_ms": (1000 * capture_stride * physics_steps_per_decision / config.PHYSICS_FPS),
        "stored_duration_values_ms": sorted(set(frame_durations_ms)),
        "expected_duration_ms": expected_duration_ms,
        "gif_total_duration_ms": gif_total_duration_ms,
        "timing_error_ms": gif_total_duration_ms - expected_duration_ms,
        "size": size,
        "decisions": decision,
        "physics_steps": int(final_info["steps"]),
        "survival_seconds": float(final_info["steps"] / config.PHYSICS_FPS),
        "score": int(final_info["score"]),
        "return": episode_return,
        "terminated": bool(final_terminated),
        "truncated": bool(final_truncated),
    }


def choose_best_result(results):
    def score(result):
        summary = result["best_summary"]
        return (
            float(summary["median_survival_seconds"]),
            float(summary["p10_survival_seconds"]),
            float(summary["mean_survival_seconds"]),
        )

    return max(results, key=score)


def create_artifacts(results, output_root, device, online_net, final_eval_seed_start, final_eval_episodes,
                      evaluate_policy_fn, deterministic_action_fn, q_values_fn, network_type, total_episodes):
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    artifact_dir = output_root / "artifacts" / f"batch_{timestamp}"
    artifact_dir.mkdir(parents=True, exist_ok=False)

    learning_curve_path = save_learning_curves(results, artifact_dir / "learning_curves.png")

    best_result = choose_best_result(results)
    source_checkpoint = Path(best_result["best_checkpoint"])
    best_checkpoint_path = artifact_dir / "best_train.pt"
    shutil.copy2(source_checkpoint, best_checkpoint_path)

    # 아티팩트 생성에 쓸 네트워크를 best 체크포인트 가중치로 복원
    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    online_net.load_state_dict(checkpoint["online_state_dict"])
    online_net.eval()

    final_seeds = list(range(final_eval_seed_start, final_eval_seed_start + final_eval_episodes))
    final_records, final_summary = evaluate_policy_fn(deterministic_action_fn, final_seeds)

    constant_action_baselines = {}
    for action, action_name in enumerate(ACTION_NAMES):
        _, baseline_summary = evaluate_policy_fn(lambda _state, fixed_action=action: fixed_action, final_seeds)
        constant_action_baselines[action_name] = baseline_summary
    best_constant_action, best_constant_summary = max(
        constant_action_baselines.items(),
        key=lambda item: (
            item[1]["median_survival_seconds"], item[1]["p10_survival_seconds"], item[1]["mean_survival_seconds"],
        ),
    )
    best_play_record = max(final_records, key=lambda record: (record["score"], record["survival_seconds"]))
    best_play_seed = int(best_play_record["seed"])

    trajectory = collect_q_trajectory(q_values_fn, best_play_seed)
    q_image_path, q_csv_path = save_q_table_visualization(
        trajectory, artifact_dir / "q_table_heatmap.png", artifact_dir / "q_values.csv"
    )
    video_metadata = save_play_gif(
        deterministic_action_fn,
        best_play_seed,
        artifact_dir / "best_model_train.gif",
        expected_physics_steps=int(trajectory["final_physics_steps"]),
        physics_steps_per_decision=int(checkpoint["env_spec"]["physics_steps_per_decision"]),
    )

    metadata = {
        "algorithm": "Dueling Double DQN" if network_type == "dueling" else "Double DQN",
        "train_seeds": [result["seed"] for result in results],
        "training_budget_episodes": total_episodes,
        "selected_train_seed": best_result["seed"],
        "selected_checkpoint_step": checkpoint["global_step"],
        "selection_summary": best_result["best_summary"],
        "final_evaluation_summary": final_summary,
        "final_evaluation_records": final_records,
        "constant_action_baselines": constant_action_baselines,
        "best_constant_action": best_constant_action,
        "best_constant_action_summary": best_constant_summary,
        "median_seconds_over_best_constant": final_summary["median_survival_seconds"]
        - best_constant_summary["median_survival_seconds"],
        "best_play_seed": best_play_seed,
        "video_selection": "highest-survival final evaluation episode; best-case showcase, not representative performance",
        "best_play_record": best_play_record,
        "q_trajectory": {
            key: value for key, value in trajectory.items() if key not in {"q_values", "actions", "physics_steps"}
        },
        "video": video_metadata,
        "artifacts": {
            "best_train": str(best_checkpoint_path),
            "learning_curves": str(learning_curve_path),
            "q_table_heatmap": str(q_image_path),
            "q_values_csv": str(q_csv_path),
            "best_model_video": str(artifact_dir / "best_model_train.gif"),
        },
    }
    save_json(artifact_dir / "artifact_metadata.json", metadata)
    save_json(
        artifact_dir / "batch_results.json",
        [
            {
                "seed": result["seed"],
                "run_dir": result["run_dir"],
                "best_checkpoint": result["best_checkpoint"],
                "best_summary": result["best_summary"],
            }
            for result in results
        ],
    )
    return metadata
