import math
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from buffer import ReplayBuffer
from model import NatureCNN


class DQNAgent:
    def __init__(
        self,
        env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        hidden_size: int,
        batch_size: int,
        discount_factor: float,
        learning_starts: int,
        train_frequency: int,
        target_network_frequency: int,
        capacity: int,
        device: str

    ):
        self.env = env
        self.device = torch.device(device)

        self.n_actions = env.n_actions              # 9
        obs_shape = env.observation_shape           # (4, 84, 84)

        self.model = NatureCNN(
            input_shape=obs_shape,
            output_size=self.n_actions,
            hidden_size=hidden_size,
        ).to(self.device)

        self.target_model = NatureCNN(
            input_shape=obs_shape,
            output_size=self.n_actions,
            hidden_size=hidden_size,
        ).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.lr = learning_rate
        self.discount_factor = discount_factor  # gamma

        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon


        self.update_count = 0
        self.learning_starts = learning_starts
        self.train_frequency = train_frequency
        self.target_network_frequency = target_network_frequency

        self.rb = ReplayBuffer(
            capacity=capacity,
            obs_shape=obs_shape,
            device=self.device,
        )
        self.batch_size = batch_size

        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.lr)

        # Track learning progress
        self.training_error = []
        self.q_values = []

    def get_action(self, obs) -> int:
        """Choose an action using epsilon-greedy strategy.

        Returns:
            action: 0 (stop), 1 (up), 2 (down), 3 (left), 4 (right),
                    5 (up-left), 6 (up-right), 7 (down-left), 8 (down-right)
        """
        # With probability epsilon: explore (random action)
        if np.random.random() < self.epsilon:
            return random.randrange(self.n_actions)

        # With probability (1-epsilon): exploit (best known action)
        else:
            with torch.no_grad():
                # (4,84,84) -> (1,4,84,84)
                obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
                return int(self.model(obs).argmax(dim=1).item())

    def update(self):
        self.update_count += 1
        if self.update_count > self.learning_starts:
            if self.update_count % self.train_frequency == 0:
                # sample from replay buffer
                data = self.rb.sample(batch_size=self.batch_size)

                # Loss = (Q(s,a) - (r+gamma*max(Q(s'))))^2
                with torch.no_grad():
                    next_q_values = self.get_next_q_values(
                        data.next_observations
                    )

                    td_target = (
                        data.rewards.flatten()
                        + self.discount_factor
                        * next_q_values
                        * (1 - data.dones.flatten())
                    )
                old_val = self.model(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Track learning progress (useful for debugging)
                self.training_error.append(loss.item())
                self.q_values.append(old_val.mean().item())

            # 되도록이면 target_network_frequency % train_frequency == 0 이 되도록 하는게 좋음
            if self.update_count % self.target_network_frequency == 0:
                self.update_target_network()

    # max (targetQ)
    def get_next_q_values(self, next_observations):
        return self.target_model(next_observations).max(dim=1).values

    def update_target_network(self):
        for target_network_param, q_network_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_network_param.data.copy_(q_network_param.data)


    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


class DDQNAgent(DQNAgent):
    # targetQ(s', argmax_a'(Q(s',a')))
    def get_next_q_values(self, next_observations):
        next_actions = self.model(next_observations).argmax(
            dim=1,
            keepdim=True,
        )
        return self.target_model(next_observations).gather(
            dim=1,
            index=next_actions,
        ).squeeze(1)


def evaluate(agent, env, num_episodes=20, seed=None, epsilon=0.0):
    """탐색/학습 없이 정책을 평가한다.

    Danmaku의 성능 지표는 reward 합이 아니라 game.state.score (= 생존 초 수)다.
    DanmakuImgEnv가 MAX_TIME_STEPS(10800 물리 스텝, score 180)에서 스스로 truncate
    하므로 여기서 인위적인 스텝 상한을 두지 않는다. 상한을 두면 목표 점수 120을
    측정할 수 없게 된다.

    agent가 None이면 random 정책을 평가한다 (baseline 비교용).
    """
    scores = []
    lengths = []
    returns = []
    action_counts = np.zeros(env.n_actions, dtype=np.int64)

    if agent is not None:
        old_epsilon = agent.epsilon
        agent.epsilon = epsilon

    rng = random.Random(seed)
    for episode in range(num_episodes):
        # 같은 시드 집합을 매번 쓰면 체크포인트 간 비교가 짝지은 비교가 된다.
        episode_seed = None if seed is None else seed + episode
        obs, info = env.reset(seed=episode_seed)
        done = False
        episode_return = 0.0
        episode_length = 0

        while not done:
            if agent is None:
                action = rng.randrange(env.n_actions)
            else:
                action = agent.get_action(obs)
            action_counts[action] += 1
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            episode_length += 1
            done = terminated or truncated

        scores.append(int(info["score"]))
        lengths.append(episode_length)
        returns.append(episode_return)

    if agent is not None:
        agent.epsilon = old_epsilon

    scores_array = np.array(scores, dtype=np.float64)
    probabilities = action_counts / max(action_counts.sum(), 1)
    nonzero = probabilities[probabilities > 0]
    entropy = max(0.0, float(-(nonzero * np.log(nonzero)).sum()))  # 최대 ln(9) = 2.197

    return {
        "num_episodes": num_episodes,
        "scores": scores,
        "mean_score": float(scores_array.mean()),
        "median_score": float(np.median(scores_array)),
        "std_score": float(scores_array.std(ddof=1)) if num_episodes > 1 else 0.0,
        "sem_score": (
            float(scores_array.std(ddof=1) / math.sqrt(num_episodes))
            if num_episodes > 1 else 0.0
        ),
        "max_score": int(scores_array.max()),
        "p_ge_120": float((scores_array >= 120).mean()),
        "p_ge_60": float((scores_array >= 60).mean()),
        "mean_length": float(np.mean(lengths)),
        "mean_return": float(np.mean(returns)),
        "action_entropy": entropy,
        "action_distribution": probabilities.tolist(),
    }

