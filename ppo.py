import copy
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


# ------------------------------------------------------------
# 1. 결과 저장 폴더
# ------------------------------------------------------------
save_dir = Path("danmaku_ppo_results")
save_dir.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. Actor-Critic Network
# ------------------------------------------------------------
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        # DQN과 같은 3개의 64-unit hidden layer를 사용한다.
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        for network in (self.actor, self.critic):
            for layer in network:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
                    nn.init.zeros_(layer.bias)

        # 초기에는 행동 확률이 한쪽으로 쏠리지 않게 한다.
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def forward(self, states):
        logits = self.actor(states)
        values = self.critic(states).squeeze(-1)
        return logits, values


# ------------------------------------------------------------
# 3. On-policy Rollout Buffer
# ------------------------------------------------------------
class RolloutBuffer:
    def __init__(self):
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.old_log_probs = []
        self.values = []
        self.next_values = []
        self.terminated = []
        self.truncated = []

    def __len__(self):
        return len(self.actions)

    def add(self, state, action, reward, old_log_prob, value, next_value, terminated, truncated):
        self.states.append(np.asarray(state, dtype=np.float32))
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.old_log_probs.append(float(old_log_prob))
        self.values.append(float(value))
        self.next_values.append(float(next_value))
        self.terminated.append(bool(terminated))
        self.truncated.append(bool(truncated))

    def compute_advantages(self, gamma, gae_lambda):
        rewards = np.asarray(self.rewards, dtype=np.float32)
        values = np.asarray(self.values, dtype=np.float32)
        next_values = np.asarray(self.next_values, dtype=np.float32)
        terminated = np.asarray(self.terminated, dtype=np.float32)
        truncated = np.asarray(self.truncated, dtype=np.float32)

        advantages = np.zeros(len(self), dtype=np.float32)
        next_advantage = 0.0

        for index in range(len(self) - 1, -1, -1):
            # DQN과 마찬가지로 실제 사망에서만 bootstrap을 제거한다.
            bootstrap_mask = 1.0 - terminated[index]
            # 서로 다른 episode의 GAE는 연결하지 않는다.
            episode_continues = (1.0 - terminated[index]) * (1.0 - truncated[index])
            delta = rewards[index] + gamma * bootstrap_mask * next_values[index] - values[index]
            next_advantage = delta + gamma * gae_lambda * episode_continues * next_advantage
            advantages[index] = next_advantage

        returns = advantages + values
        return advantages, returns


# ------------------------------------------------------------
# 4. 정책 및 reward helper
# ------------------------------------------------------------
def policy(state, deterministic=False):
    state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        logits, value = main_net(state_tensor)
        distribution = Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else distribution.sample()
        log_prob = distribution.log_prob(action)
    return int(action.item()), float(log_prob.item()), float(value.item())


def state_value(state):
    state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        _logits, value = main_net(state_tensor)
    return float(value.item())


def distance_reward(target_env=None):
    target_env = env if target_env is None else target_env
    game_state = target_env.game.state
    if len(game_state.balls) == 0:
        return 0.0

    agent = game_state.agent
    distance_min = min(
        math.hypot(ball.x - agent.x, ball.y - agent.y) - ball.r - agent.r
        for ball in game_state.balls
    )
    return float(np.clip(distance_min / distance_pixels, 0.0, 1.0))


def current_entropy_coef(global_step):
    fraction = min(global_step / total_steps, 1.0)
    return entropy_coef_start + fraction * (entropy_coef_end - entropy_coef_start)


def current_learning_rate(global_step):
    # 마지막에도 초기 learning rate의 10%는 유지한다.
    remaining = max(0.1, 1.0 - global_step / total_steps)
    return learning_rate * remaining


# ------------------------------------------------------------
# 5. PPO update
# ------------------------------------------------------------
def update_parameter_with_loss():
    advantages, returns = rollout.compute_advantages(gamma, gae_lambda)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    states = torch.as_tensor(np.asarray(rollout.states), dtype=torch.float32, device=device)
    actions = torch.as_tensor(rollout.actions, dtype=torch.long, device=device)
    old_log_probs = torch.as_tensor(rollout.old_log_probs, dtype=torch.float32, device=device)
    advantages = torch.as_tensor(advantages, dtype=torch.float32, device=device)
    returns = torch.as_tensor(returns, dtype=torch.float32, device=device)

    for group in optimizer.param_groups:
        group["lr"] = current_learning_rate(global_step)

    entropy_coef = current_entropy_coef(global_step)
    metric_values = {
        key: []
        for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "grad_norm")
    }

    early_stop_kl = False
    epochs_completed = 0

    for _epoch in range(update_epochs):
        indices = torch.randperm(len(rollout), device=device)

        for start in range(0, len(rollout), batch_size):
            batch_indices = indices[start : start + batch_size]

            logits, values = main_net(states[batch_indices])
            distribution = Categorical(logits=logits)
            new_log_probs = distribution.log_prob(actions[batch_indices])
            entropy = distribution.entropy().mean()

            log_ratio = new_log_probs - old_log_probs[batch_indices]
            ratio = log_ratio.exp()
            unclipped_objective = ratio * advantages[batch_indices]
            clipped_objective = (
                torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
                * advantages[batch_indices]
            )
            policy_loss = -torch.minimum(unclipped_objective, clipped_objective).mean()

            # DQN과 동일하게 이상치에 덜 민감한 Smooth L1을 사용한다.
            value_loss = F.smooth_l1_loss(values, returns[batch_indices])
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(main_net.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = ((ratio - 1.0).abs() > clip_ratio).float().mean()

            values_to_add = (
                policy_loss,
                value_loss,
                entropy,
                approx_kl,
                clip_fraction,
                grad_norm,
            )
            for key, value in zip(metric_values, values_to_add):
                metric_values[key].append(float(value.item()))

            # 한 rollout에서 정책이 지나치게 변하면 남은 epoch를 중단한다.
            if approx_kl.item() > target_kl:
                early_stop_kl = True
                break

        epochs_completed += 1
        if early_stop_kl:
            break

    rollout.clear()
    result = {key: float(np.mean(values)) for key, values in metric_values.items()}
    result.update(
        {
            "learning_rate": current_learning_rate(global_step),
            "entropy_coef": entropy_coef,
            "epochs_completed": epochs_completed,
            "early_stop_kl": early_stop_kl,
        }
    )
    return result


# ------------------------------------------------------------
# 6. 평가
# ------------------------------------------------------------
def summarize_evaluation(records):
    survival = np.asarray([record["survival_seconds"] for record in records], dtype=np.float64)
    rewards = np.asarray([record["reward"] for record in records], dtype=np.float64)
    return {
        "episodes": len(records),
        "completion_rate": float(np.mean([record["truncated"] for record in records])),
        "mean_survival_seconds": float(survival.mean()),
        "median_survival_seconds": float(np.median(survival)),
        "p10_survival_seconds": float(np.percentile(survival, 10)),
        "min_survival_seconds": float(survival.min()),
        "max_survival_seconds": float(survival.max()),
        "mean_reward": float(rewards.mean()),
    }


def evaluation_score(summary):
    return (
        float(summary["completion_rate"]),
        float(summary["median_survival_seconds"]),
        float(summary["p10_survival_seconds"]),
        float(summary["mean_survival_seconds"]),
    )


def evaluate(n_eval=100, max_time_steps=None, seed_base=1_000_000):
    eval_env = DanmakuVecEnv()
    if max_time_steps is not None:
        eval_env.max_time_steps = max_time_steps

    records = []
    for eval_seed in range(n_eval):
        state, _ = eval_env.reset(seed=seed_base + eval_seed)
        total_reward = 0.0

        while True:
            action, _log_prob, _value = policy(state, deterministic=True)
            state, reward, terminated, truncated, info = eval_env.step(action)
            total_reward += float(reward)

            if terminated or truncated:
                records.append(
                    {
                        "seed": seed_base + eval_seed,
                        "reward": total_reward,
                        "physics_steps": int(info["steps"]),
                        "survival_seconds": float(info["steps"] / config.PHYSICS_FPS),
                        "score": int(info["score"]),
                        "truncated": bool(truncated),
                    }
                )
                break

    return records, summarize_evaluation(records)


def save_video(path, seed=1_000_000):
    return ppo_visualize.save_play_video(
        lambda state: policy(state, deterministic=True)[0],
        path,
        seed=seed,
    )


def save_training_artifacts():
    ppo_visualize.save_learning_curve(
        episode_history,
        eval_history,
        update_history,
        save_dir / "learning_curve.png",
    )
    ppo_visualize.save_score_plot(episode_history, save_dir / "score.png")
    ppo_visualize.save_success_rate_plot(
        episode_successes,
        eval_history,
        save_dir / "success_rate.png",
    )
    ppo_visualize.save_survival_time_plot(
        episode_history,
        eval_history,
        save_dir / "survival_time.png",
    )
    ppo_visualize.save_loss_plot(update_history, save_dir / "loss.png")
    ppo_visualize.save_action_distribution_plot(
        action_counts,
        save_dir / "action_distribution.png",
    )


# ------------------------------------------------------------
# 7. 하이퍼파라미터
# ------------------------------------------------------------
train_seed = 44
random.seed(train_seed)
np.random.seed(train_seed)
torch.manual_seed(train_seed)

total_steps = 20_000_000
rollout_steps = 4096
update_epochs = 4
batch_size = 256
learning_rate = 0.0003
gamma = 0.99
gae_lambda = 0.95
clip_ratio = 0.2
value_coef = 0.5
entropy_coef_start = 0.02
entropy_coef_end = 0.002
max_grad_norm = 0.5
target_kl = 0.03

# DQN과 동일한 reward 설정
death_penalty = -3.0
truncated_reward = 10.0
distance_pixels = 100
distance_ratio = 0.1
stall_penalty = 0.05

# DQN과 동일한 curriculum 및 step 주기
curriculum_stages = [600 * index for index in range(1, 13)]
curriculum_success_threshold = 0.5
eval_every_steps = 50_000
render_every_steps = 500_000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------
# 8. 학습 초기화
# ------------------------------------------------------------
env = DanmakuVecEnv()
curriculum_index = 0
env.max_time_steps = curriculum_stages[curriculum_index]

state_dim = env.observation_shape[0]
action_dim = len(env.action_space)
main_net = ActorCritic(state_dim, action_dim).to(device)
optimizer = optim.Adam(main_net.parameters(), lr=learning_rate, eps=1e-5)
rollout = RolloutBuffer()

global_step = 0
episode = 0
next_eval_step = eval_every_steps
next_render_step = render_every_steps

action_counts = [0] * action_dim
episode_rewards = []
episode_successes = []
episode_steps = []
episode_history = []
eval_history = []
update_history = []

latest_eval_summary = None
latest_update_metrics = {
    key: float("nan")
    for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "grad_norm")
}
best_eval_score = (-math.inf, -math.inf, -math.inf, -math.inf)
best_state_dict = None

log_path = save_dir / "training_log.csv"
log_file, log_writer = ppo_visualize.open_training_log(log_path)
run_bar = tqdm(total=total_steps, desc="[PPO run 0]", unit="step")


# ------------------------------------------------------------
# 9. Step 기반 PPO 학습
# ------------------------------------------------------------
while global_step < total_steps:
    episode += 1
    state, _ = env.reset(seed=train_seed * 1_000_000 + episode)

    total_reward = 0.0
    episode_learning_reward = 0.0
    used_steps = 0
    env_terminated = False
    truncated = False

    while not (env_terminated or truncated) and global_step < total_steps:
        action, log_prob, value = policy(state)
        action_counts[action] += 1

        previous_dist = distance_reward()
        previous_agent_x = env.game.state.agent.x
        previous_agent_y = env.game.state.agent.y

        next_state, reward, env_terminated, truncated, info = env.step(action)
        finished = env_terminated or truncated

        next_dist = 0.0 if finished else distance_reward()
        agent = env.game.state.agent
        moved = (agent.x != previous_agent_x) or (agent.y != previous_agent_y)

        dist_reward = distance_ratio * (gamma * next_dist - previous_dist)
        if env_terminated:
            base_reward = death_penalty
        elif truncated:
            base_reward = truncated_reward
        else:
            base_reward = float(reward)

        stall = (not moved) and (previous_dist < 1.0)
        learning_reward = base_reward + dist_reward - (stall_penalty if stall else 0.0)

        next_value = 0.0 if env_terminated else state_value(next_state)
        rollout.add(
            state,
            action,
            learning_reward,
            log_prob,
            value,
            next_value,
            env_terminated,
            truncated,
        )

        state = next_state
        total_reward += float(reward)
        episode_learning_reward += learning_reward
        used_steps += 1
        global_step += 1
        run_bar.update(1)

        if len(rollout) >= rollout_steps:
            latest_update_metrics = update_parameter_with_loss()
            update_history.append({"global_step": global_step, **latest_update_metrics})

    success = 1 if truncated else 0
    survival_seconds = info["steps"] / config.PHYSICS_FPS

    episode_rewards.append(total_reward)
    episode_successes.append(success)
    episode_steps.append(used_steps)
    episode_history.append(
        {
            "reward": total_reward,
            "score": info["score"],
            "survival_seconds": survival_seconds,
        }
    )

    ppo_visualize.log_episode(
        log_writer,
        log_file,
        {
            "run": 0,
            "episode": episode,
            "global_step": global_step,
            "reward": total_reward,
            "learning_reward": episode_learning_reward,
            "success": success,
            "decisions": used_steps,
            "physics_steps": int(info["steps"]),
            "survival_seconds": survival_seconds,
            "eval_median_survival_seconds": (
                "" if latest_eval_summary is None else latest_eval_summary["median_survival_seconds"]
            ),
            **latest_update_metrics,
        },
    )

    # DQN과 마찬가지로 global step 기준으로 평가하고 승급한다.
    if global_step >= next_eval_step:
        _records, latest_eval_summary = evaluate(
            n_eval=100,
            max_time_steps=env.max_time_steps,
        )
        eval_history.append({"global_step": global_step, **latest_eval_summary})

        score = evaluation_score(latest_eval_summary)
        if score > best_eval_score:
            best_eval_score = score
            best_state_dict = copy.deepcopy(main_net.state_dict())
            ppo_visualize.save_checkpoint(best_state_dict, save_dir / "best_model.pt")

        if (
            latest_eval_summary["completion_rate"] >= curriculum_success_threshold
            and curriculum_index < len(curriculum_stages) - 1
        ):
            # 서로 다른 curriculum 단계의 on-policy rollout을 섞지 않는다.
            if len(rollout) > 0:
                latest_update_metrics = update_parameter_with_loss()
                update_history.append({"global_step": global_step, **latest_update_metrics})

            curriculum_index += 1
            env.max_time_steps = curriculum_stages[curriculum_index]
            best_eval_score = (-math.inf, -math.inf, -math.inf, -math.inf)

        while next_eval_step <= global_step:
            next_eval_step += eval_every_steps

    if global_step >= next_render_step:
        ppo_visualize.save_checkpoint(main_net.state_dict(), save_dir / "latest_model.pt")
        ppo_visualize.save_checkpoint(
            {
                "model": main_net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "global_step": global_step,
                "episode": episode,
                "curriculum_index": curriculum_index,
            },
            save_dir / "latest_training_state.pt",
        )
        save_training_artifacts()
        save_video(save_dir / f"progress_step_{global_step}.mp4")

        while next_render_step <= global_step:
            next_render_step += render_every_steps

    run_bar.set_postfix(
        success=success,
        balls=len(env.game.state.balls),
        stage=f"{curriculum_index + 1}/{len(curriculum_stages)}({env.max_time_steps})",
        eval=(
            "-"
            if latest_eval_summary is None
            else f"{latest_eval_summary['completion_rate']:.2f}/{latest_eval_summary['mean_survival_seconds']:.1f}s"
        ),
    )


# ------------------------------------------------------------
# 10. 최종 저장 및 최종 목표 평가
# ------------------------------------------------------------
if len(rollout) > 0:
    latest_update_metrics = update_parameter_with_loss()
    update_history.append({"global_step": global_step, **latest_update_metrics})

run_bar.close()
log_file.close()
ppo_visualize.save_checkpoint(main_net.state_dict(), save_dir / "latest_model.pt")

# latest와 현재 curriculum best를 진짜 최종 목표(120초)에서 비교한다.
candidate_states = [("latest", copy.deepcopy(main_net.state_dict()))]
if best_state_dict is not None:
    candidate_states.append(("best_stage", best_state_dict))

candidate_results = []
for label, state_dict in candidate_states:
    main_net.load_state_dict(state_dict)
    _records, summary = evaluate(
        n_eval=100,
        max_time_steps=curriculum_stages[-1],
        seed_base=1_500_000,
    )
    candidate_results.append((evaluation_score(summary), label, copy.deepcopy(state_dict), summary))

candidate_results.sort(reverse=True, key=lambda row: row[0])
_score, selected_label, selected_state_dict, validation_summary = candidate_results[0]
main_net.load_state_dict(selected_state_dict)
ppo_visualize.save_checkpoint(selected_state_dict, save_dir / "best_model.pt")

test_records, final_eval_summary = evaluate(
    n_eval=300,
    max_time_steps=curriculum_stages[-1],
    seed_base=2_000_000,
)
ppo_visualize.save_json(
    save_dir / "eval_summary.json",
    {
        "selected_checkpoint": selected_label,
        "validation": validation_summary,
        "test": final_eval_summary,
    },
)

save_training_artifacts()
ppo_visualize.save_reward_curve(
    [episode_rewards],
    save_dir / "reward_curve.png",
    title="PPO — Danmaku-RL",
)

best_record = max(test_records, key=lambda record: record["survival_seconds"])
save_video(save_dir / "best_model.mp4", seed=best_record["seed"])
