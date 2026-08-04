from collections import deque
from tqdm import tqdm
import numpy as np
import random
import config 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from env import DanmakuVecEnv
from pathlib import Path
import dqn_visualize


# ------------------------------------------------------------
# 1. 결과 저장 폴더
# ------------------------------------------------------------

save_dir = Path("danmaku_ddqn_results")
save_dir.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 2. Network update
# ------------------------------------------------------------

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64), 
            nn.ReLU(),
            nn.Linear(64, 64), 
            nn.ReLU(), 
            nn.Linear(64, 64), 
            nn.ReLU(), 
            nn.Linear(64, action_dim)
            )  # state, dim

    def forward(self, x):
        return self.network(x)

def policy(state, eps):
    if random.random() < eps:
        action = random.choice(env.action_space)
    else:
        state_tensor = torch.as_tensor(state, dtype = torch.float32, device = device).unsqueeze(0)
        q_value = main_net(state_tensor).detach().cpu().numpy().flatten()
        max_actions = np.flatnonzero(q_value == q_value.max())
        action = int(np.random.choice(max_actions)) # 동점이면 무작위 선택 (특정 방향 편향 방지)
    return action


def update_parameter_with_loss():
    experiences = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*experiences)

    states = torch.tensor( np.array(states), dtype=torch.float32, device=device )
    next_states = torch.tensor( np.array(next_states), dtype=torch.float32, device=device )
    actions = torch.tensor( actions, dtype=torch.long, device=device )
    rewards = torch.tensor( rewards, dtype=torch.float32, device=device )
    dones = torch.tensor( dones, dtype=torch.bool, device=device )

    q_values = main_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_actions = main_net(next_states).argmax(dim=1, keepdim=True)
        next_q_values = target_net(next_states).gather(1, next_actions).squeeze(1)
        target_q_values = rewards + gamma * next_q_values * (1.0 - dones.float())

    # loss = F.mse_loss(q_values, target_q_values)
    loss = F.smooth_l1_loss(q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def synchronize_target_net():
    with torch.no_grad():
        target_net.load_state_dict(main_net.state_dict())

# 새 시드로 평가
def evaluate(n_eval=100, max_time_steps=None):
    eval_env = DanmakuVecEnv()
    if max_time_steps is not None:
        eval_env.max_time_steps = max_time_steps
    survived = 0
    eval_max_steps = eval_env.max_time_steps // config.N_FRAME_SKIP + 1
    survival_seconds = []

    for eval_seed in range(n_eval):
        state, _ = eval_env.reset(seed = 1_000_000 + eval_seed)
        for _ in range(eval_max_steps):
            state, reward, term, trun, info = eval_env.step(policy(state, 0.0))
            if term or trun:
                survived += int(trun)
                survival_seconds.append(info["steps"] / config.PHYSICS_FPS)
                break
    return survived / n_eval, float(np.mean(survival_seconds))

def find_best_seed(n_eval=100, max_time_steps=None):
    eval_env = DanmakuVecEnv()
    if max_time_steps is not None:
        eval_env.max_time_steps = max_time_steps
    eval_max_steps = eval_env.max_time_steps // config.N_FRAME_SKIP + 1
    best_seed, best_survival = 1_000_000, -1.0

    for eval_seed in range(n_eval):
        seed = 1_000_000 + eval_seed
        state, _ = eval_env.reset(seed=seed)
        for _ in range(eval_max_steps):
            state, reward, term, trun, info = eval_env.step(policy(state, 0.0))
            if term or trun:
                survival = info["steps"] / config.PHYSICS_FPS
                if survival > best_survival:
                    best_survival, best_seed = survival, seed
                break
    return best_seed

def distance_reward():
    state = env.game.state
    if len(state.balls) == 0:
        return 0.0
    agent = state.agent 
    distance_min = min(((ball.x-agent.x)**2 + (ball.y-agent.y)**2) ** 0.5 - ball.r - agent.r for ball in state.balls)
    return float(np.clip(distance_min/distance_pixels , 0.0, 1.0))


# ------------------------------------------------------------
# 4. 파라미터 설정
# ------------------------------------------------------------
 
train_seed = 44
random.seed(train_seed)
np.random.seed(train_seed)
torch.manual_seed(train_seed)

n_runs = 1
render_every = 1000
episodes = 100000
eps_decay = 0.9997
eps_end = 0.0005
batch_size = 64
learning_rate = 0.0005
gamma = 0.99
target_net_update_interval = 200
distance_pixels = 100 # 거리가 100 이내이면 패널티 보상
distance_ratio = 0.1 # 기존 reward에 섞어줄 거리 패널티 비율
truncated_reward = 10 # 커리큘럼 목표(현재 단계) 완주 시 보너스
stall_penalty = 0.05 # 위험한데 안 움직이면 깎을 페널티
curriculum_stages = [300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600] # 공 5개(300스텝)씩 늘려가는 MAX_TIME_STEPS 커리큘럼
curriculum_success_threshold = 0.70 # eval success_rate가 이 이상이면 다음 단계로 승급

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 시각화 & 로그 저장용
all_runs_episode_rewards = []
all_runs_episode_successes = []
all_runs_episode_steps = []
all_runs_episode_losses = []
all_runs_eval_history = []
all_runs_action_counts = []

eval_every = 500

log_path = save_dir / "training_log.csv"
log_file, log_writer = dqn_visualize.open_training_log(log_path)


# ------------------------------------------------------------
# 4. 학습 
# ------------------------------------------------------------

for run_id in range(n_runs):
    env = DanmakuVecEnv()
    curriculum_index = 0
    env.max_time_steps = curriculum_stages[curriculum_index]
    state_dim = env.observation_shape[0]
    action_dim = len(env.action_space)
    memory = deque(maxlen=100000)
    eps = 1

    action_counts = [0] * action_dim

    main_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(main_net.state_dict())
    optimizer = optim.Adam(main_net.parameters(), lr=learning_rate)
    global_step = 0


    # 로그
    episode_rewards = []
    episode_scores = []
    episode_survival_seconds = []
    episode_successes = []
    episode_steps = []
    episode_losses = []
    episode_epsilons = []

    eval_history = []
    latest_eval_success_rate = 0.0
    latest_eval_mean_survival = 0.0
    best_eval_score = (-1.0, -1.0)
    best_state_dict = None

    run_bar = tqdm(range(1, episodes + 1), desc=f"[run {run_id}]")

    # episode 진행
    for episode in run_bar:
        state, _ = env.reset(seed=(train_seed + run_id) * 1_000_000 + episode)

        total_reward = 0.0
        losses_in_episode = []
        used_steps = 0

        # 한 판 진행 스텝
        for movement in range(1, config.MAX_TIME_STEPS // config.N_FRAME_SKIP + 2):
            action = policy(state, eps)
            action_counts[action] += 1

            previous_dist = distance_reward()
            previous_agent_x = env.game.state.agent.x
            previous_agent_y = env.game.state.agent.y
            next_state, reward, env_terminated, truncated, info = env.step(action)

            finished = env_terminated or truncated 

            next_dist = 0.0 if finished else distance_reward()
            agent = env.game.state.agent
            moved = (agent.x != previous_agent_x) or (agent.y != previous_agent_y)
            # learning_reward = reward
            dist_reward = distance_ratio * (gamma * next_dist - previous_dist) # potential based reward shaping
            #	공에서 멀어졌으면: next_dist > previous_dist라서 dist_reward가 양수 → 실질적으로 보너스
		    #	공에 가까워졌으면: next_dist < previous_dist라서 dist_reward가 음수 → base_reward에 음수를 더하는 거니까 실질적으로 페널티		
            # if env_terminated:
            #     learning_reward = -1

            if env_terminated:
                base_reward = -3
            elif truncated:
                base_reward = truncated_reward
            else:
                base_reward = reward
            stall = (not moved) and (previous_dist < 1.0) # 에이전트 주위에 공이 있어 위험한 상황이었는데 + 움직이지 않았을 경우
            learning_reward = base_reward + dist_reward - (stall_penalty if stall else 0.0) # 추가적으로 0.05 패널티 차감 

            memory.append( (state, action, learning_reward, next_state, env_terminated) )

            state = next_state
            final_state = next_state
            total_reward += reward                          # 그래프/성공률 판단은 순수 환경 보상 기준
            used_steps = movement

            if len(memory) >= batch_size:
                current_loss = update_parameter_with_loss()
                losses_in_episode.append(current_loss)

                global_step += 1
                if global_step % target_net_update_interval == 0:
                    synchronize_target_net()

            if finished:
                break

        success = 1 if truncated else 0

        mean_loss = np.mean(losses_in_episode) if losses_in_episode else np.nan

        episode_rewards.append(total_reward) # 0 or 1
        episode_scores.append(info["score"])
        episode_survival_seconds.append(info["steps"] / config.PHYSICS_FPS)
        episode_successes.append(success)
        episode_steps.append(used_steps)
        episode_losses.append(mean_loss)

        eps = max(eps * eps_decay, eps_end)

        if episode % eval_every == 0:
            latest_eval_success_rate, latest_eval_mean_survival = evaluate(max_time_steps=env.max_time_steps)
            eval_history.append((episode, latest_eval_success_rate, latest_eval_mean_survival))

            eval_score = (latest_eval_success_rate, latest_eval_mean_survival)
            if eval_score > best_eval_score:
                best_eval_score = eval_score
                best_state_dict = {k: v.clone() for k, v in main_net.state_dict().items()}

            if latest_eval_success_rate >= curriculum_success_threshold and curriculum_index < len(curriculum_stages) - 1:
                curriculum_index += 1
                env.max_time_steps = curriculum_stages[curriculum_index]
                best_eval_score = (-1.0, -1.0)  # 난이도가 바뀌었으니 best 비교 기준 리셋

        if episode % render_every == 0:
            dqn_visualize.save_play_video(
                lambda s: policy(s, 0.0), save_dir / f"progress_run{run_id}_ep{episode}.mp4"
            )

        run_bar.set_postfix(
            success=success, balls=len(env.game.state.balls), eps=f"{eps:.3f}",
            stage=f"{curriculum_index + 1}/{len(curriculum_stages)}({env.max_time_steps})",
            eval=f"{latest_eval_success_rate:.2f}/{latest_eval_mean_survival:.1f}s",
        )

        dqn_visualize.log_episode(
            log_writer, log_file, run_id, episode, total_reward, success, used_steps, mean_loss, eps,
            latest_eval_success_rate, latest_eval_mean_survival,
        )

    dqn_visualize.save_best_model(main_net.state_dict(), save_dir / f"latest_model_run{run_id}.pt")

    if best_state_dict is not None:
        main_net.load_state_dict(best_state_dict)  # 이 run에서 평가 성공률이 가장 높았던 시점으로 복원
        dqn_visualize.save_best_model(best_state_dict, save_dir / f"best_model_run{run_id}.pt")

        # 커리큘럼 진행 정도와 무관하게, 최종 목표(curriculum_stages[-1]) 기준으로 진짜 실력을 평가
        final_success_rate, final_mean_survival = evaluate(max_time_steps=curriculum_stages[-1])
        dqn_visualize.save_eval_summary(
            final_success_rate, final_mean_survival, save_dir / f"eval_summary_run{run_id}.txt"
        )
        dqn_visualize.save_learning_curve(
            episode_rewards, episode_successes, eval_history, save_dir / f"learning_curve_run{run_id}.png"
        )
        dqn_visualize.save_score_plot(episode_scores, save_dir / f"score_run{run_id}.png")
        dqn_visualize.save_success_rate_plot(episode_successes, eval_history, save_dir / f"success_rate_run{run_id}.png")
        dqn_visualize.save_survival_time_plot(episode_survival_seconds, eval_history, save_dir / f"survival_time_run{run_id}.png")
        dqn_visualize.save_loss_plot(episode_losses, save_dir / f"loss_run{run_id}.png")
        dqn_visualize.save_action_distribution_plot(action_counts, save_dir / f"action_distribution_run{run_id}.png")
        dqn_visualize.save_play_video(
            lambda s: policy(s, 0.0), save_dir / f"best_model_run{run_id}.mp4",
            seed=find_best_seed(max_time_steps=curriculum_stages[-1]),
        )

    all_runs_episode_rewards.append(episode_rewards)
    all_runs_episode_successes.append(episode_successes)
    all_runs_episode_steps.append(episode_steps)
    all_runs_episode_losses.append(episode_losses)
    all_runs_eval_history.append(eval_history)
    all_runs_action_counts.append(action_counts)

log_file.close()
dqn_visualize.save_reward_curve(all_runs_episode_rewards, save_dir / "reward_curve.png", title="DDQN — Danmaku-RL")