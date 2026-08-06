import argparse
import math
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import save_dqn

import config
from env import DanmakuVecEnv

ACTION_NAMES = ("STOP", "UP", "DOWN", "LEFT", "RIGHT", "UP_LEFT", "UP_RIGHT", "DOWN_LEFT", "DOWN_RIGHT")


def resolve_device(device_name):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device_name)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return resolved


def parse_args():
    parser = argparse.ArgumentParser(description="Train Dueling Double DQN directly on DanmakuVecEnv.")
    parser.add_argument("--seed", type=int, default=700140)
    parser.add_argument("--total-episodes", type=int, default=1000000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--final-eval-episodes", type=int, default=500)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", type=Path, default=THIS_DIR / "episode_1000000_outputs")
    parser.add_argument("--demo-path", type=Path, default=THIS_DIR / "human_demos.npz")
    parser.add_argument("--demo-fraction", type=float, default=0.2)
    return parser.parse_args()


# ------------------------------------------------------------
# 1. Network
# ------------------------------------------------------------

class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x):
        return self.network(x)


class DuelingQNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_dim):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), 
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU()
        )
        self.value = nn.Linear(hidden_dim, 1) # mean or max 값 
        self.advantage = nn.Linear(hidden_dim, n_actions)

    def forward(self, x):
        features = self.features(x)
        value = self.value(features)
        advantage = self.advantage(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True) 


def build_network(obs_dim, n_actions):
    network = DuelingQNetwork if network_type == "dueling" else QNetwork
    return network(obs_dim, n_actions, hidden_dim).to(device)


# ------------------------------------------------------------
# 2. 정책 / 리플레이 버퍼 / 학습 함수
# ------------------------------------------------------------

def policy(state, eps):
    if action_rng.random() < eps:
        action = int(action_rng.integers(action_dim))
    else:
        action = deterministic_action(state)
    return action


@torch.inference_mode()
def deterministic_action(state): # exploitation
    state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    q_value = online_net(state_tensor).squeeze(0)
    return int(q_value.argmax().item())


@torch.inference_mode()
def q_values(state): # for visualize 
    state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    return online_net(state_tensor).squeeze(0).cpu().numpy()


def replay_add(state, action, reward, next_state, terminated, truncated):
    global replay_position, replay_size
    replay_observations[replay_position] = state
    replay_next_observations[replay_position] = next_state
    replay_actions[replay_position] = action
    replay_rewards[replay_position] = reward
    replay_terminated[replay_position] = terminated
    replay_truncated[replay_position] = truncated

    replay_position = (replay_position + 1) % replay_capacity
    replay_size = min(replay_size + 1, replay_capacity)


def replay_sample(size):
    demo_count = int(round(size * demo_fraction)) if demo_size > 0 else 0
    replay_count = size - demo_count
    indices = replay_rng.integers(0, replay_size, size=replay_count)

    observations = replay_observations[indices]
    next_observations = replay_next_observations[indices]
    actions = replay_actions[indices]
    rewards = replay_rewards[indices]
    terminated = replay_terminated[indices]
    truncated = replay_truncated[indices]

    if demo_count > 0:
        # 사람 플레이 경험은 회전(circular)되지 않는 영구 버퍼 — 매 배치마다 일정 비율로 항상 섞임 (DQfD 방식)
        demo_indices = replay_rng.integers(0, demo_size, size=demo_count)
        observations = np.concatenate([observations, demo_observations[demo_indices]])
        next_observations = np.concatenate([next_observations, demo_next_observations[demo_indices]])
        actions = np.concatenate([actions, demo_actions[demo_indices]])
        rewards = np.concatenate([rewards, demo_rewards[demo_indices]])
        terminated = np.concatenate([terminated, demo_terminated[demo_indices]])
        truncated = np.concatenate([truncated, demo_truncated[demo_indices]])

    return {
        "states": torch.as_tensor(observations, dtype=torch.float32, device=device),
        "actions": torch.as_tensor(actions, dtype=torch.long, device=device),
        "rewards": torch.as_tensor(rewards, dtype=torch.float32, device=device),
        "next_states": torch.as_tensor(next_observations, dtype=torch.float32, device=device),
        "terminated": torch.as_tensor(terminated, dtype=torch.float32, device=device),
        "truncated": torch.as_tensor(truncated, dtype=torch.float32, device=device),
    }


def update_parameter_with_loss():
    batch = replay_sample(batch_size) # 버퍼에서 배치개수만큼 추출

    all_q_values = online_net(batch["states"]) # online net 에 넣어 Q(s, a) 구하기 
    selected_q_values = all_q_values.gather(1, batch["actions"].unsqueeze(1)).squeeze(1)  # 실제로 취한 행동의 q 값 

    with torch.no_grad():
        next_actions = online_net(batch["next_states"]).argmax(dim=1, keepdim=True) # argmax로 다음 상태에서 어떤 행동을 고를지 
        next_q_values = target_net(batch["next_states"]).gather(1, next_actions).squeeze(1) # 다음 상태를 target net에 넣고 평가 
        done = torch.maximum(batch["terminated"], batch["truncated"]) 
        target_q_values = batch["rewards"] + gamma * next_q_values * (1.0 - done)

    loss = F.smooth_l1_loss(selected_q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online_net.parameters(), max_norm)
    optimizer.step()

    return loss.item(), selected_q_values.mean().item()


def synchronize_target_net():
    with torch.no_grad():
        target_net.load_state_dict(online_net.state_dict())


def epsilon_by_episode(episode_index):
    fraction = min(max(episode_index / eps_decay_episodes, 0.0), 1.0)
    return eps_start + fraction * (eps_end - eps_start)


def distance_reward():
    state = env.game.state
    if not state.balls:
        return 0.0
    agent = state.agent
    distance_min = min(
        ((ball.x - agent.x) ** 2 + (ball.y - agent.y) ** 2) ** 0.5 - ball.r - agent.r for ball in state.balls
    )
    return float(np.clip(distance_min / distance_pixels, 0.0, 1.0))


def summarize_episodes(records):
    if not records:
        return {
            "episodes": 0,
            "mean_survival_seconds": 0.0,
            "median_survival_seconds": 0.0,
            "p10_survival_seconds": 0.0,
            "completion_rate": 0.0,
            "mean_dominant_action_fraction": 0.0,
            "mean_boundary_fraction": 0.0,
            "mean_stationary_fraction": 0.0,
            "action_fractions": [],
        }

    survival = np.asarray([record["survival_seconds"] for record in records], dtype=np.float64)
    scores = np.asarray([record["score"] for record in records], dtype=np.float64)
    dominant_action_fractions = np.asarray(
        [record.get("dominant_action_fraction", 0.0) for record in records], dtype=np.float64
    )
    boundary_fractions = np.asarray([record.get("boundary_fraction", 0.0) for record in records], dtype=np.float64)
    stationary_fractions = np.asarray([record.get("stationary_fraction", 0.0) for record in records], dtype=np.float64)
    action_counts = np.sum(np.asarray([record.get("action_counts", []) for record in records], dtype=np.int64), axis=0)
    summary = {
        "episodes": len(records),
        "mean_survival_seconds": float(survival.mean()),
        "median_survival_seconds": float(np.median(survival)),
        "std_survival_seconds": float(survival.std()),
        "p10_survival_seconds": float(np.percentile(survival, 10)),
        "min_survival_seconds": float(survival.min()),
        "max_survival_seconds": float(survival.max()),
        "mean_score": float(scores.mean()),
        "median_score": float(np.median(scores)),
        "p10_score": float(np.percentile(scores, 10)),
        "mean_dominant_action_fraction": float(dominant_action_fractions.mean()),
        "mean_boundary_fraction": float(boundary_fractions.mean()),
        "mean_stationary_fraction": float(stationary_fractions.mean()),
        "action_fractions": action_counts / max(int(action_counts.sum()), 1),
        "completion_rate": float(np.mean([bool(record["truncated"]) for record in records])),
    }
    for seconds in (30, 60, 90, 120, 150, 180):
        summary[f"survival_rate_{seconds}s"] = float(np.mean(survival >= seconds))
    return summary


def evaluate_policy(action_fn, eval_seeds):
    eval_env = DanmakuVecEnv()
    records = []

    for seed in eval_seeds:
        observation, _ = eval_env.reset(seed=seed)
        episode_return = 0.0
        action_counts = np.zeros(len(tuple(eval_env.action_space)), dtype=np.int64)
        boundary_decisions = 0
        stationary_decisions = 0

        while True:
            action = int(action_fn(observation))
            action_counts[action] += 1
            previous_agent_x = eval_env.game.state.agent.x
            previous_agent_y = eval_env.game.state.agent.y
            (observation, reward, terminated, truncated, info) = eval_env.step(action)
            episode_return += float(reward)
            agent = eval_env.game.state.agent
            stationary_decisions += int(
                math.isclose(agent.x, previous_agent_x) and math.isclose(agent.y, previous_agent_y)
            )
            on_x_boundary = agent.x <= agent.r + 1e-6 or agent.x >= config.SCREEN_WIDTH - agent.r - 1e-6
            on_y_boundary = agent.y <= agent.r + 1e-6 or agent.y >= config.SCREEN_HEIGHT - agent.r - 1e-6
            boundary_decisions += int(on_x_boundary or on_y_boundary)

            if terminated or truncated:
                records.append(
                    {
                        "seed": int(seed),
                        "return": episode_return,
                        "physics_steps": int(info["steps"]),
                        "survival_seconds": float(info["steps"] / config.PHYSICS_FPS),
                        "score": int(info["score"]),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "action_counts": action_counts.copy(),
                        "dominant_action_fraction": float(action_counts.max() / action_counts.sum()),
                        "boundary_fraction": float(boundary_decisions / action_counts.sum()),
                        "stationary_fraction": float(stationary_decisions / action_counts.sum()),
                    }
                )
                break

    return records, summarize_episodes(records)


# 새 시드로 평가 (evaluate_policy가 median/p10/생존율 등 요약 통계를 계산)
def run_evaluation(n_eval, eval_seed_start): #100
    was_training = online_net.training
    online_net.eval()
    eval_seeds = list(range(eval_seed_start, eval_seed_start + n_eval))
    records, summary = evaluate_policy(deterministic_action, eval_seeds)
    if was_training:
        online_net.train()
    return records, summary


# ------------------------------------------------------------
# 5. 파라미터 설정
# ------------------------------------------------------------

args = parse_args()

train_seed = args.seed
network_type = "dueling"  # "mlp" 또는 "dueling"
learning_rate = 1e-4
gamma = 0.95
gradient_steps = 1
eps_start = 1.0
eps_end = 0.05
survival_reward = 0.01  # 매 스텝 생존 시 기본 보상
collision_penalty = -1.0  # 공에 맞아 죽었을 때 페널티
completion_reward = 1.0  # 제한 시간까지 생존(truncated) 시 보너스
distance_pixels = 150.0  # 가장 가까운 공까지 이 거리(px) 이내면 위험 구간으로 간주
distance_ratio = 0.1  # 기존 reward에 섞어줄 clearance shaping 비율
stall_penalty = 0.002  # 위험한데(공 존재) 안 움직이면 깎을 페널티
max_norm = 10.0  # gradient clipping norm 상한 (loss 폭주로 인한 Q-value 발산 방지)
eval_seed_start = 100_000
final_eval_seed_start = 200_000

total_episodes = args.total_episodes
hidden_dim = 256
replay_capacity = 500_000
learning_starts = 2000
batch_size = 512
train_frequency = 8
target_net_update_interval = 2000
eps_decay_episodes = 100000
log_interval = 10_000
eval_interval_episodes = 2_50
checkpoint_interval_episodes = 5_00
progress_save_interval_steps = 10000  # 이 스텝마다 체크포인트 + 학습 곡선을 중간 저장
gif_save_interval_steps = 100000  # 이 스텝마다 고정 시드로 플레이 GIF를 중간 저장 (에피소드 롤아웃이 필요해 더 무거움)
eval_episodes = args.eval_episodes
final_eval_episodes = args.final_eval_episodes
demo_path = args.demo_path
demo_fraction = args.demo_fraction

HYPERPARAMETERS = {
    "total_episodes": total_episodes,
    "hidden_dim": hidden_dim,
    "network_type": network_type,
    "learning_rate": learning_rate,
    "gamma": gamma,
    "replay_capacity": replay_capacity,
    "learning_starts": learning_starts,
    "batch_size": batch_size,
    "train_frequency": train_frequency,
    "gradient_steps": gradient_steps,
    "target_net_update_interval": target_net_update_interval,
    "eps_start": eps_start,
    "eps_end": eps_end,
    "eps_decay_episodes": eps_decay_episodes,
    "survival_reward": survival_reward,
    "collision_penalty": collision_penalty,
    "completion_reward": completion_reward,
    "distance_pixels": distance_pixels,
    "distance_ratio": distance_ratio,
    "stall_penalty": stall_penalty,
    "max_norm": max_norm,
    "demo_fraction": demo_fraction,
}

if demo_path.exists():
    demo_data = np.load(demo_path)
    demo_observations = demo_data["observations"]
    demo_next_observations = demo_data["next_observations"]
    demo_actions = demo_data["actions"]
    demo_rewards = demo_data["rewards"]
    demo_terminated = demo_data["terminated"]
    demo_truncated = demo_data["truncated"]
    demo_size = len(demo_actions)
    print(f"loaded {demo_size} human demo transitions from {demo_path}", flush=True)
else:
    demo_size = 0
    print(f"no human demo file at {demo_path}; training without a demo buffer", flush=True)

device = resolve_device(args.device)
output_root = args.output_root
output_root.mkdir(parents=True, exist_ok=True)

torch.set_float32_matmul_precision("high")


# ------------------------------------------------------------
# 3. 학습 (여러 시드 반복)
# ------------------------------------------------------------

seeds = [train_seed]
results = []

for seed in seeds: # train seed, 현재는 1개 
    run_dir = save_dqn.make_run_dir(output_root / "runs", "dueling_dqn", seed)

    save_dqn.set_global_seed(seed)
    action_rng = np.random.default_rng(seed + 10_000)
    replay_rng = np.random.default_rng(seed + 20_000)

    env = DanmakuVecEnv()
    env_spec = save_dqn.validate_vec_env(env, seed)
    obs_dim = env_spec["obs_dim"]
    action_dim = env_spec["n_actions"]

    if demo_size > 0 and demo_observations.shape[1] != obs_dim:
        raise ValueError(f"demo observation dim {demo_observations.shape[1]} != env obs_dim {obs_dim}")

    replay_observations = np.empty((replay_capacity, obs_dim), dtype=np.float32)
    replay_next_observations = np.empty((replay_capacity, obs_dim), dtype=np.float32)
    replay_actions = np.empty(replay_capacity, dtype=np.int64)
    replay_rewards = np.empty(replay_capacity, dtype=np.float32)
    replay_terminated = np.empty(replay_capacity, dtype=np.bool_)
    replay_truncated = np.empty(replay_capacity, dtype=np.bool_)
    replay_position = 0
    replay_size = 0

    online_net = build_network(obs_dim, action_dim) 
    target_net = build_network(obs_dim, action_dim)
    target_net.load_state_dict(online_net.state_dict())
    target_net.requires_grad_(False)
    target_net.eval()
    optimizer = torch.optim.Adam(online_net.parameters(), lr=learning_rate)

    print(
        f"[seed {seed}] device={device} obs={obs_dim} actions={action_dim} episodes={total_episodes:,}",
        flush=True,
    )

    global_step = 0
    total_physics_steps = 0
    best_score = (-float("inf"), -float("inf"), -float("inf"))
    best_summary = None
    best_checkpoint = run_dir / "best_model.pt"
    latest_checkpoint = run_dir / "latest_model.pt"
    metrics_path = run_dir / "metrics.jsonl"

    episode_history = []
    evaluation_history = []
    recent_losses = deque(maxlen=1_000)
    recent_q_values = deque(maxlen=1_000)
    last_log_time = time.perf_counter()
    next_progress_step = progress_save_interval_steps
    next_gif_step = gif_save_interval_steps
    preview_seed = 900_000 + seed  # 항상 같은 시드로 찍어서 학습이 진행될수록 행동이 어떻게 변하는지 비교

    save_dqn.save_json(
        run_dir / "config.json",
        {
            "algorithm": "Dueling Double DQN" if network_type == "dueling" else "Double DQN",
            "seed": seed,
            "hyperparameters": HYPERPARAMETERS,
            "env": env_spec,
            "action_names": ACTION_NAMES,
        },
    )

    # 학습 시작 전 초기 평가 (0 에피소드 시점 기준선)
    evaluation_record, best_score, best_summary = save_dqn.log_evaluation(
        run_evaluation, eval_episodes, eval_seed_start, seed, run_dir, metrics_path, 0, global_step, total_physics_steps,
        best_checkpoint, best_score, best_summary, env_spec,
        network_type, HYPERPARAMETERS, online_net, target_net, optimizer, action_rng,
    )
    evaluation_history.append(evaluation_record)

    run_bar = tqdm(range(1, total_episodes + 1), desc=f"[seed {seed}]")

    # episode 진행
    for episode in run_bar:
        episode_index = episode - 1
        epsilon = epsilon_by_episode(episode_index)
        state, _ = env.reset(seed=seed * 1_000_000 + episode_index)

        environment_return = 0.0
        training_return = 0.0
        action_counts = np.zeros(action_dim, dtype=np.int64)
        decisions = 0

        # 한 판 진행 스텝
        while True:
            action = policy(state, epsilon)
            action_counts[action] += 1

            previous_physics_steps = env.game.state.steps
            previous_agent_x = env.game.state.agent.x
            previous_agent_y = env.game.state.agent.y
            previous_dist = distance_reward()

            next_state, reward, terminated, truncated, info = env.step(action)
            finished = terminated or truncated

            next_dist = 0.0 if finished else distance_reward()
            agent = env.game.state.agent
            moved = (agent.x != previous_agent_x) or (agent.y != previous_agent_y)

            if terminated:
                base_reward = collision_penalty
            elif truncated:
                base_reward = completion_reward
            else:
                base_reward = survival_reward
            dist_reward = distance_ratio * (gamma * next_dist - previous_dist)  # potential based reward shaping
            stall = bool(env.game.state.balls) and not moved  # 공이 있는데 움직이지 않음
            learning_reward = base_reward + dist_reward - (stall_penalty if stall else 0.0)

            replay_add(state, action, learning_reward, next_state, terminated, truncated)

            total_physics_steps += max(int(info["steps"] - previous_physics_steps), 0)
            global_step += 1
            environment_return += float(reward)
            training_return += learning_reward
            decisions += 1
            state = next_state

            if replay_size >= learning_starts and global_step % train_frequency == 0:
                step_losses, step_qs = [], []
                for _ in range(gradient_steps):
                    loss, mean_q = update_parameter_with_loss()
                    step_losses.append(loss)
                    step_qs.append(mean_q)
                recent_losses.append(float(np.mean(step_losses)))
                recent_q_values.append(float(np.mean(step_qs)))

            if global_step >= learning_starts and global_step % target_net_update_interval == 0:
                synchronize_target_net()

            if global_step % log_interval == 0:
                now = time.perf_counter()
                elapsed = max(now - last_log_time, 1e-8)
                steps_per_second = log_interval / elapsed
                last_log_time = now

                recent_episodes = episode_history[-500:]
                mean_seconds = (
                    float(np.mean([record["survival_seconds"] for record in recent_episodes]))
                    if recent_episodes else 0.0
                )
                save_dqn.append_jsonl(metrics_path, {
                    "type": "progress",
                    "train_seed": seed,
                    "global_step": global_step,
                    "episode_index": episode_index,
                    "total_physics_steps": total_physics_steps,
                    "epsilon": epsilon,
                    "replay_size": replay_size,
                    "recent_mean_survival_seconds": mean_seconds,
                    "recent_mean_loss": float(np.mean(recent_losses)) if recent_losses else 0.0,
                    "recent_mean_q": float(np.mean(recent_q_values)) if recent_q_values else 0.0,
                    "steps_per_second": steps_per_second,
                })

            if global_step >= next_progress_step:
                # 최신 체크포인트와 학습 곡선만 가볍게 중간 저장
                save_dqn.save_checkpoint(latest_checkpoint, seed, global_step, total_physics_steps, episode_index, env_spec, best_summary,
                                          network_type, HYPERPARAMETERS, online_net, target_net, optimizer, action_rng)
                progress_result = {"seed": seed, "episode_history": episode_history, "evaluation_history": evaluation_history}
                save_dqn.save_learning_curves([progress_result], run_dir / "learning_curve_progress.png")
                while next_progress_step <= global_step:
                    next_progress_step += progress_save_interval_steps

            if global_step >= next_gif_step:
                # 고정 시드 1개를 끝까지 플레이해서 현재 정책의 GIF를 남김
                preview_trajectory = save_dqn.collect_q_trajectory(q_values, preview_seed)
                save_dqn.save_play_gif(
                    deterministic_action,
                    preview_seed,
                    run_dir / f"progress_step{global_step}.gif",
                    expected_physics_steps=int(preview_trajectory["final_physics_steps"]),
                    physics_steps_per_decision=env_spec["physics_steps_per_decision"],
                )
                while next_gif_step <= global_step:
                    next_gif_step += gif_save_interval_steps

            if finished:
                break

        dominant_action_fraction = float(action_counts.max() / decisions)

        episode_record = {
            "type": "episode",
            "train_seed": seed,
            "episode_index": episode_index,
            "global_step": global_step,
            "episode_seed": seed * 1_000_000 + episode_index,
            "environment_return": environment_return,
            "training_return": training_return,
            "score": int(info["score"]),
            "physics_steps": int(info["steps"]),
            "decisions": decisions,
            "action_counts": action_counts.copy(),
            "dominant_action_fraction": dominant_action_fraction,
            "survival_seconds": float(info["steps"] / config.PHYSICS_FPS),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "epsilon": epsilon,
        }
        episode_history.append(episode_record)
        save_dqn.append_jsonl(metrics_path, episode_record)

        if episode % eval_interval_episodes == 0:
            evaluation_record, best_score, best_summary = save_dqn.log_evaluation(
                run_evaluation, eval_episodes, eval_seed_start, seed, run_dir, metrics_path, episode, global_step, total_physics_steps,
                best_checkpoint, best_score, best_summary, env_spec,
                network_type, HYPERPARAMETERS, online_net, target_net, optimizer, action_rng,
            )
            evaluation_history.append(evaluation_record)

        if episode % checkpoint_interval_episodes == 0:
            save_dqn.save_checkpoint(latest_checkpoint, seed, global_step, total_physics_steps, episode, env_spec, best_summary,
                                      network_type, HYPERPARAMETERS, online_net, target_net, optimizer, action_rng)

        run_bar.set_postfix(
            score=info["score"],
            survival=f"{info['steps'] / config.PHYSICS_FPS:.1f}s",
            eps=f"{epsilon:.3f}",
            best=f"{best_summary['median_survival_seconds']:.1f}s" if best_summary else "-",
        )

    if evaluation_history[-1]["episode_index"] != total_episodes:
        evaluation_record, best_score, best_summary = save_dqn.log_evaluation(
            run_evaluation, eval_episodes, eval_seed_start, seed, run_dir, metrics_path, total_episodes, global_step, total_physics_steps,
            best_checkpoint, best_score, best_summary, env_spec,
            network_type, HYPERPARAMETERS, online_net, target_net, optimizer, action_rng,
        )
        evaluation_history.append(evaluation_record)

    save_dqn.save_checkpoint(latest_checkpoint, seed, global_step, total_physics_steps, total_episodes, env_spec, best_summary,
                              network_type, HYPERPARAMETERS, online_net, target_net, optimizer, action_rng)

    result = {
        "seed": seed,
        "run_dir": str(run_dir),
        "env_spec": env_spec,
        "best_checkpoint": str(best_checkpoint),
        "latest_checkpoint": str(latest_checkpoint),
        "best_summary": best_summary,
        "episode_history": episode_history,
        "evaluation_history": evaluation_history,
    }
    save_dqn.save_json(run_dir / "run_result.json", result)
    results.append(result)


# ------------------------------------------------------------
# 4. 최종 아티팩트
# ------------------------------------------------------------

artifacts = save_dqn.create_artifacts(
    results, output_root, device, online_net, final_eval_seed_start, final_eval_episodes,
    evaluate_policy, deterministic_action, q_values, network_type, total_episodes,
)
print("Training complete.", {"seed": train_seed, "artifacts": artifacts["artifacts"]}, flush=True)
