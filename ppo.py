import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm

import config
import ppo_visualize
from env import DanmakuVecEnv


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor = nn.Sequential( # 정책 신경망, 각 행동의 확률(logits)
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )
        self.critic = nn.Sequential(  # 상태 가치, 행동을 고르지 않고 actor 학습을 위한 baseline 역할 
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(), 
            nn.Linear(64, 1)
        )

        # 가중치 초기화: 가중치 행렬을 무작위 직교행렬로 채우는 초기화 방법
        for network in (self.actor, self.critic):
            for layer in network:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain = math.sqrt(2))
                    nn.init.zeros_(layer.bias) # bias = 0 
        # 마지막 층만 덮어쓰기(출력층을 작은 값으로 초기화 시켜서 actor가 골고루 행동을 고르도록 함)
        nn.init.orthogonal_(self.actor[-1].weight, gain = 0.01)
        nn.init.zeros_(self.actor[-1].bias)
        nn.init.orthogonal_(self.critic[-1].weight, gain= 0.01)
        nn.init.zeros_(self.critic[-1].bias)


    def forward(self, x):
        logits = self.actor(x) # 각 선택에 대한 확률 
        value = self.critic(x).squeeze(-1) # 가치 
        return logits, value 
    


class RolloutBuffer: # PPO는 현재 정책으로 플레이하며 경험 모은 후 그걸로 학습 
    def __init__(self):
        self.clear()

    def clear(self): # 버퍼 초기화 
        self.states = []
        self.actions = []
        self.rewards = []
        self.old_log_probs = []
        self.values = []
        self.next_values = []
        self.terminated = []
        self.truncated = []

    # 버퍼에 추가하는 함수 
    def add(self, state, action, reward, old_log_prob, value, next_value, terminated, truncated):
        self.states.append(np.asarray(state, dtype=np.float32))
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.old_log_probs.append(float(old_log_prob))
        self.values.append(float(value))
        self.next_values.append(float(next_value))
        self.terminated.append(bool(terminated))
        self.truncated.append(bool(truncated))

    # 버퍼에 몇 스텝 모였는지 
    def __len__(self):
        return len(self.actions)

    # GAE 계산 
    # advantage = action 이 평균보다 얼마나 더 좋았는가 
    def compute_advantages(self, gamma, gae_lambda):
        rewards = np.asarray(self.rewards, dtype=np.float32)
        values = np.asarray(self.values, dtype=np.float32)
        next_values = np.asarray(self.next_values, dtype=np.float32)
        terminated = np.asarray(self.terminated, dtype=np.float32)
        truncated = np.asarray(self.truncated, dtype=np.float32)
        advantages = np.zeros(len(self), dtype=np.float32)
        next_advantage = 0.0

        for index in range(len(self) - 1, -1, -1): # 역순 계산 
            bootstrap_mask = 1.0 - terminated[index] # terminated 와 truncated를 구분하기 위함 
            episode_continues = (1.0 - terminated[index]) * (1.0 - truncated[index]) # 게임이 계속 진행되고 있는지 
            delta = rewards[index] + gamma * bootstrap_mask * next_values[index] - values[index] # TD error 
            next_advantage = delta + gamma * gae_lambda * episode_continues * next_advantage # delta 를 누적 
            advantages[index] = next_advantage

        returns = advantages + values # critic 의 학습 목표 
        return advantages, returns

# 행동을 결정하는 policy 
def policy(state, deterministic=False):
    state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0) # (state_dim) -> (1, state_dim)
    with torch.no_grad(): # 행동선택 
        logits, value = main_net(state_tensor)
        distribution = Categorical(logits=logits) # logits -> softmax -> 이산 확률분포 
        action = logits.argmax(dim=-1) if deterministic else distribution.sample() # 가장 높은 확률 선택 
        log_prob = distribution.log_prob(action)
    return int(action.item()), float(log_prob.item()), float(value.item())


def state_value(state): # 다음 상태의 가치를 구하는 함수
    state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        _, value = main_net(state_tensor)
    return float(value.item())


def distance_reward():
    state = env.game.state
    if len(state.balls) == 0:
        return 0.0
    agent = state.agent
    distance_min = min(((ball.x-agent.x)**2 + (ball.y-agent.y)**2) ** 0.5 - ball.r - agent.r for ball in state.balls)
    return float(np.clip(distance_min/distance_pixels, 0.0, 1.0))


# ------------------------------------------------------------
# 1. 결과 저장 폴더
# ------------------------------------------------------------
save_dir = Path("danmaku_ppo_results")
save_dir.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. Network update
# ------------------------------------------------------------
def update_parameter_with_loss(): # rollout으로 actor와 critic을 업데이트 
    advantages, returns = rollout.compute_advantages(gamma, gae_lambda) # GAE Advantage 와 critic target return 계산 
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) # 정규화 

    states = torch.as_tensor(np.array(rollout.states), dtype=torch.float32, device=device)
    actions = torch.as_tensor(rollout.actions, dtype=torch.long, device=device)
    old_log_probs = torch.as_tensor(rollout.old_log_probs, dtype=torch.float32, device=device)
    advantages = torch.as_tensor(advantages, dtype=torch.float32, device=device)
    returns = torch.as_tensor(returns, dtype=torch.float32, device=device)

    metric_values = {key: [] for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "grad_norm")}

    for _ in range(update_epochs): # epoch 
        indices = torch.randperm(len(rollout), device=device)
        for start in range(0, len(rollout), batch_size): # mini batch 
            batch_indices = indices[start:start + batch_size]

            logits, values = main_net(states[batch_indices]) # mini batch에서 logits 와 value 가져옴 
            distribution = Categorical(logits=logits) # 새 행동 분포 
            new_log_probs = distribution.log_prob(actions[batch_indices]) 
            entropy = distribution.entropy().mean() # 정책 분포의 평균 entropy 계산 (entropy 가 크면 행동 골고루 선택)

            log_ratio = new_log_probs - old_log_probs[batch_indices]
            ratio = log_ratio.exp()
            unclipped_objective = ratio * advantages[batch_indices] # 정책 목적 함수 
            clipped_objective = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages[batch_indices] # ratio 를 [0.8, 1.2]로 제한 
            policy_loss = -torch.min(unclipped_objective, clipped_objective).mean() # 둘 중 작은 값 선택 -> 음수 
            value_loss = F.mse_loss(values, returns[batch_indices]) #critic 예측 values와 GAE return 사이의 MSE 계산 
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy #전체 loss 결합 

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(main_net.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = ((ratio - 1.0).abs() > clip_ratio).float().mean()
            metric_values["policy_loss"].append(float(policy_loss.item()))
            metric_values["value_loss"].append(float(value_loss.item()))
            metric_values["entropy"].append(float(entropy.item()))
            metric_values["approx_kl"].append(float(approx_kl.item()))
            metric_values["clip_fraction"].append(float(clip_fraction.item()))
            metric_values["grad_norm"].append(float(grad_norm.item()))

    rollout.clear() #rollout buffer 비움 
    return {key: float(np.mean(values)) for key, values in metric_values.items()}


def summarize_evaluation(records):
    survival = np.asarray([record["survival_seconds"] for record in records], dtype=np.float64)
    rewards = np.asarray([record["reward"] for record in records], dtype=np.float64)
    return {
        "episodes": len(records),
        "mean_survival_seconds": float(survival.mean()),
        "median_survival_seconds": float(np.median(survival)),
        "p10_survival_seconds": float(np.percentile(survival, 10)),
        "min_survival_seconds": float(survival.min()),
        "max_survival_seconds": float(survival.max()),
        "mean_reward": float(rewards.mean()),
        "completion_rate": float(np.mean([record["truncated"] for record in records])),
    }


def evaluation_score(summary):
    return (
        float(summary["median_survival_seconds"]),
        float(summary["p10_survival_seconds"]),
        float(summary["completion_rate"]),
        float(summary["mean_survival_seconds"]),
    )


# 새 시드로 평가
def evaluate(n_eval=20):
    eval_env = DanmakuVecEnv()
    eval_max_steps = config.MAX_TIME_STEPS // config.N_FRAME_SKIP + 1
    records = []

    for eval_seed in range(n_eval):
        state, _ = eval_env.reset(seed=1_000_000 + eval_seed)
        total_reward = 0.0

        for _ in range(eval_max_steps):
            action, _log_prob, _value = policy(state, deterministic=True)
            state, reward, term, trun, info = eval_env.step(action)
            total_reward += float(reward)
            if term or trun:
                records.append(
                    {
                        "reward": total_reward,
                        "physics_steps": int(info["steps"]),
                        "survival_seconds": float(info["steps"] / config.PHYSICS_FPS),
                        "score": int(info["score"]),
                        "truncated": bool(trun),
                    }
                )
                break

    return records, summarize_evaluation(records)


def save_video_from_eval(path):
    ppo_visualize.save_play_video(lambda s: policy(s, deterministic=True)[0], path, seed=1_000_000)


# ------------------------------------------------------------
# 4. 학습
# ------------------------------------------------------------

# 파라미터 설정
train_seed = 44
random.seed(train_seed)
np.random.seed(train_seed)
torch.manual_seed(train_seed)

n_runs = 1

episodes = 5000
rollout_steps = 2048
update_epochs = 10
batch_size = 64
learning_rate = 0.0005
gamma = 0.99
gae_lambda = 0.95
clip_ratio = 0.2
value_coef = 0.5
entropy_coef = 0.01
max_grad_norm = 0.5
death_penalty = -1.0
distance_pixels = 50  # 거리가 50 이내이면 패널티 보상
distance_ratio = 0.1  # 기존 reward에 섞어줄 거리 패널티 비율


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 시각화 & 로그 저장용
all_runs_episode_rewards = []
all_runs_episode_successes = []
all_runs_episode_steps = []
all_runs_eval_history = []
all_runs_action_counts = []

eval_every = 100
render_every = 500

log_path = save_dir / "training_log.csv"
log_file, log_writer = ppo_visualize.open_training_log(log_path)


# 학습
for run_id in range(n_runs):
    env = DanmakuVecEnv()
    state_dim = env.observation_shape[0]
    action_dim = len(env.action_space)
    rollout = RolloutBuffer()

    action_counts = [0] * action_dim

    main_net = ActorCritic(state_dim, action_dim).to(device)
    optimizer = optim.Adam(main_net.parameters(), lr=learning_rate)
    global_step = 0

    # 로그
    episode_rewards = []
    episode_successes = []
    episode_steps = []
    episode_history = []
    eval_history = []
    update_history = []

    latest_eval_summary = None
    latest_update_metrics = {key: float("nan") for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "grad_norm")}
    best_eval_score = (-math.inf, -math.inf, -math.inf, -math.inf)
    best_state_dict = None

    run_bar = tqdm(range(1, episodes + 1), desc=f"[PPO run {run_id}]")

    # episode 진행
    for episode in run_bar:
        state, _ = env.reset(seed=(train_seed + run_id) * 1_000_000 + episode)

        total_reward = 0.0
        episode_learning_reward = 0.0
        used_steps = 0

        # 한 판 진행 스텝
        for movement in range(1, config.MAX_TIME_STEPS // config.N_FRAME_SKIP + 2):
            action, log_prob, value = policy(state) # Main net 통과 -> actor 행동 선택, critic 가치 획득 
            action_counts[action] += 1

            previous_dist = distance_reward()
            next_state, reward, env_terminated, truncated, info = env.step(action) # 한 번 스텝 수행

            finished = env_terminated or truncated

            next_dist = 0.0 if finished else distance_reward()
            dist_reward = distance_ratio * (gamma * next_dist - previous_dist)

            base_reward = death_penalty if env_terminated else reward
            learning_reward = base_reward + dist_reward

            next_value = 0.0 if env_terminated else state_value(next_state)
            rollout.add(state, action, learning_reward, log_prob, value, next_value, env_terminated, truncated) # 버퍼에 추가 

            state = next_state
            total_reward += reward
            episode_learning_reward += learning_reward
            used_steps = movement
            global_step += 1

            if len(rollout) >= rollout_steps: # rollout step 이상이 되면 업데이트(PPO는 DQN처럼 한 스텝당 한 번 업데이트 되는 것이 아님) 
                latest_update_metrics = update_parameter_with_loss()
                update_history.append({"global_step": global_step, **latest_update_metrics})

            if finished:
                break

        success = 1 if truncated else 0
        survival_seconds = info["steps"] / config.PHYSICS_FPS

        episode_rewards.append(total_reward)
        episode_successes.append(success)
        episode_steps.append(used_steps)
        episode_history.append({"reward": total_reward, "survival_seconds": survival_seconds})

        if episode % eval_every == 0:
            _records, latest_eval_summary = evaluate()
            eval_history.append({"global_step": global_step, **latest_eval_summary})

            if evaluation_score(latest_eval_summary) > best_eval_score:
                best_eval_score = evaluation_score(latest_eval_summary)
                best_state_dict = {k: v.clone() for k, v in main_net.state_dict().items()}

        if episode % render_every == 0:
            save_video_from_eval(save_dir / f"progress_run{run_id}_ep{episode}.mp4")

        run_bar.set_postfix(
            success=success,
            balls=len(env.game.state.balls),
            policy_loss=f"{latest_update_metrics['policy_loss']:.3f}",
            eval="-" if latest_eval_summary is None else f"{latest_eval_summary['median_survival_seconds']:.1f}s",
        )

        ppo_visualize.log_episode(
            log_writer,
            log_file,
            {
                "run": run_id,
                "episode": episode,
                "global_step": global_step,
                "reward": total_reward,
                "learning_reward": episode_learning_reward,
                "success": success,
                "decisions": used_steps,
                "physics_steps": int(info["steps"]),
                "survival_seconds": survival_seconds,
                "eval_median_survival_seconds": "" if latest_eval_summary is None else latest_eval_summary["median_survival_seconds"],
                **latest_update_metrics,
            },
        )

    if best_state_dict is not None:
        main_net.load_state_dict(best_state_dict)  # 이 run에서 평가 성공률이 가장 높았던 시점으로 복원
        ppo_visualize.save_checkpoint(best_state_dict, save_dir / f"best_model_run{run_id}.pt")
        ppo_visualize.save_learning_curve(
            episode_history, eval_history, update_history, save_dir / f"learning_curve_run{run_id}.png"
        )
        save_video_from_eval(save_dir / f"best_model_run{run_id}.mp4")

    all_runs_episode_rewards.append(episode_rewards)
    all_runs_episode_successes.append(episode_successes)
    all_runs_episode_steps.append(episode_steps)
    all_runs_eval_history.append(eval_history)
    all_runs_action_counts.append(action_counts)

log_file.close()
ppo_visualize.save_reward_curve(all_runs_episode_rewards, save_dir / "reward_curve.png", title="PPO — Danmaku-RL")
