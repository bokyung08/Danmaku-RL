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
save_dir = Path("danmaku_dqn_results")
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
    states_onehot = torch.tensor(np.array(states), dtype = torch.float32, device = device)
    next_states = torch.tensor(np.array(next_states), dtype = torch.float32, device = device)

    actions = torch.tensor( actions, dtype=torch.long, device=device )
    rewards = torch.tensor( rewards, dtype=torch.float32, device=device )
    dones = torch.tensor( dones, dtype=torch.bool, device=device )

    q_values = main_net(states_onehot) .gather(1, actions.unsqueeze(1)) .squeeze(1)

    with torch.no_grad():
        next_q_values = target_net( next_states ).max(dim=1).values
        target_q_values = rewards + gamma * next_q_values * (1.0 - dones.float())

    loss = F.mse_loss(q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def synchronize_target_net():
    with torch.no_grad():
        target_net.load_state_dict(main_net.state_dict())

# 새 시드로 평가
def evaluate(n_eval=20):
    eval_env = DanmakuVecEnv()
    survived = 0
    eval_max_steps = config.MAX_TIME_STEPS // config.N_FRAME_SKIP + 1
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
# ------------------------------------------------------------
# 4. 파라미터 설정
# ------------------------------------------------------------
 
train_seed = 44
random.seed(train_seed)
np.random.seed(train_seed)
torch.manual_seed(train_seed)

n_runs = 1

episodes = 5000
eps_decay = 0.999
eps_end = 0.0005
batch_size = 64
learning_rate = 0.0005
gamma = 0.99
target_net_update_interval = 200


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 시각화 & 로그 저장용
all_runs_episode_rewards = []
all_runs_episode_successes = []
all_runs_episode_steps = []
all_runs_episode_losses = []
all_runs_eval_history = []
all_runs_action_counts = []

qmap_snapshot_interval = 100000  # global_step 기준, 이 스텝마다 Q-value 히트맵 저장

eval_every = 100
render_every = 500

log_path = save_dir / "training_log.csv"
log_file, log_writer = dqn_visualize.open_training_log(log_path)


# ------------------------------------------------------------
# 4. 학습 
# ------------------------------------------------------------

for run_id in range(n_runs):
    env = DanmakuVecEnv()
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
        final_state = state
        used_steps = 0

        # 한 판 진행 스텝
        for movement in range(1, config.MAX_TIME_STEPS // config.N_FRAME_SKIP + 2):
            action = policy(state, eps)
            action_counts[action] += 1

            next_state, reward, env_terminated, truncated, _ = env.step(action)

            finished = env_terminated or truncated 

            learning_reward = reward
            if env_terminated:
                learning_reward = -1

            memory.append( (state, action, learning_reward, next_state, finished) )

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

        episode_rewards.append(total_reward) #0 or 1
        episode_successes.append(success)
        episode_steps.append(used_steps)
        episode_losses.append(mean_loss)
        episode_epsilons.append(eps)

        eps = max(eps * eps_decay, eps_end)

        if episode % eval_every == 0:
            latest_eval_success_rate, latest_eval_mean_survival = evaluate()
            eval_history.append((episode, latest_eval_success_rate, latest_eval_mean_survival))

            eval_score = (latest_eval_success_rate, latest_eval_mean_survival)
            if eval_score > best_eval_score:
                best_eval_score = eval_score
                best_state_dict = {k: v.clone() for k, v in main_net.state_dict().items()}

        if episode % render_every == 0:
            dqn_visualize.save_play_gif(
                lambda s: policy(s, 0.0), save_dir / f"progress_run{run_id}_ep{episode}.gif"
            )

        run_bar.set_postfix(
            success=success, reward=f"{total_reward:.1f}", eps=f"{eps:.3f}",
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
        dqn_visualize.save_learning_curve(
            episode_rewards, episode_successes, eval_history, save_dir / f"learning_curve_run{run_id}.png"
        )
        dqn_visualize.save_play_gif(lambda s: policy(s, 0.0), save_dir / f"best_model_run{run_id}.gif")

    all_runs_episode_rewards.append(episode_rewards)
    all_runs_episode_successes.append(episode_successes)
    all_runs_episode_steps.append(episode_steps)
    all_runs_episode_losses.append(episode_losses)
    all_runs_eval_history.append(eval_history)
    all_runs_action_counts.append(action_counts)

log_file.close()
dqn_visualize.save_reward_curve(all_runs_episode_rewards, save_dir / "reward_curve.png", title="DQN — Danmaku-RL")