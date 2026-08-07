import math
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from buffer import ReplayBuffer
from data_augmentation import augment_transitions
from model import NatureCNN, MLP, AttentionQNetwork


class DQNAgent:
    def __init__(
        self,
        env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        hidden_size: int,
        layer_norm: bool,
        batch_size: int,
        discount_factor: float,
        learning_starts: int,
        train_frequency: int,
        target_network_frequency: int,
        capacity: int,
        device: str,
        loss_fn: str = "huber",
        max_grad_norm: float = 10.0,
        dueling_net: bool = False,
        use_attention: bool = False,
        num_heads: int = 4,
        attention_fusion: str = "residual",
        attention_position_mode: str = "relative",

    ):
        self.env = env
        self.device = torch.device(device)

        self.n_actions = env.n_actions              # 9
        obs_shape = env.observation_shape           # (4, 84, 84)

        # len(obs_shape) > 2 -> NatureCNN
        if len(obs_shape) > 2:
            model_type = NatureCNN
            extra_kwargs = {}
        elif use_attention:
            model_type = AttentionQNetwork
            extra_kwargs = {  # attention을 위한 hyperparameter
                "num_heads": num_heads,
                "fusion_mode": attention_fusion,
                "position_mode": attention_position_mode,
            }
        else:
            model_type = MLP
            extra_kwargs = {}

        self.model = model_type(
            input_shape=obs_shape,
            output_size=self.n_actions,
            hidden_size=hidden_size,
            layer_norm=layer_norm,
            dueling_net=dueling_net,
            **extra_kwargs,
        ).to(self.device)

        self.target_model = model_type(
            input_shape=obs_shape,
            output_size=self.n_actions,
            hidden_size=hidden_size,
            layer_norm=layer_norm,
            dueling_net=dueling_net,
            **extra_kwargs,
        ).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.lr = learning_rate
        self.discount_factor = discount_factor  # gamma
        self.loss_fn = loss_fn
        self.max_grad_norm = max_grad_norm

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
        """
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

    def update(self, data_augmentation=False, augmentation_mode="spawn_safe"):
        self.update_count += 1
        if self.update_count > self.learning_starts:
            if self.update_count % self.train_frequency == 0:
                # sample from replay buffer
                data = self.rb.sample(batch_size=self.batch_size)
                obs, actions, rewards, next_obs, dones = data

                if data_augmentation:
                    obs, actions, next_obs = augment_transitions(
                        obs, actions, next_obs, augmentation_mode
                    )


                # Loss = (Q(s,a) - (r+gamma*max(Q(s'))))^2
                with torch.no_grad():
                    next_q_values = self.get_next_q_values(next_obs)

                    td_target = (
                        rewards.flatten()
                        + self.discount_factor
                        * next_q_values
                        * (1 - dones.flatten())
                    )
                old_val = self.model(obs).gather(1, actions).squeeze(1)
                if self.loss_fn == "huber":
                    loss = F.smooth_l1_loss(td_target, old_val)
                else:
                    loss = F.mse_loss(td_target, old_val)

                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm is not None and self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
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


class PQNAgent:
    """PQN(Parallelised Q-Network)
    replay buffer/target network 없이, num_envs개 환경으로 모은 rollout(num_steps 길이)으로 
    Q(lambda) return을 계산해 학습. 
    https://docs.cleanrl.dev/rl-algorithms/pqn/ (cleanrl의 pqn_atari_envpool.py)
    """

    def __init__(
        self,
        env,
        num_envs: int,
        num_steps: int,
        learning_rate: float,
        gamma: float,
        q_lambda: float,
        hidden_size: int,
        layer_norm: bool,
        device: str,
        dueling_net: bool = False,
        loss_fn: str = "huber",
        max_grad_norm: float = 10.0,
        use_attention: bool = False,
        num_heads: int = 4,
        attention_fusion: str = "residual",
        attention_position_mode: str = "relative",
    ):
        self.device = torch.device(device)
        self.n_actions = env.n_actions
        obs_shape = env.observation_shape

        if len(obs_shape) > 2:
            model_type, extra_kwargs = NatureCNN, {}
        elif use_attention:
            model_type, extra_kwargs = AttentionQNetwork, {
                "num_heads": num_heads,
                "fusion_mode": attention_fusion,
                "position_mode": attention_position_mode,
            }
        else:
            model_type, extra_kwargs = MLP, {}

        self.model = model_type(
            input_shape=obs_shape,
            output_size=self.n_actions,
            hidden_size=hidden_size,
            layer_norm=layer_norm,
            dueling_net=dueling_net,
            **extra_kwargs,
        ).to(self.device)
        # target network가 따로 없다; save_checkpoint()가 기대하는 인터페이스를
        # 맞추려고 같은 모델을 가리키게 해둔다.
        self.target_model = self.model

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.q_lambda = q_lambda
        self.loss_fn = loss_fn
        self.max_grad_norm = max_grad_norm
        self.num_envs = num_envs
        self.num_steps = num_steps
        # evaluate()가 참조하는 속성. rollout 수집 중에는 act()에 epsilon을 직접
        # 넘기므로(선형 스케줄이라 값이 매 step 바뀜) 이 값은 쓰이지 않는다.
        self.epsilon = 0.0

        # rollout 저장 공간 (iteration 하나 분량). terminated(진짜 죽음)와
        # truncated(시간제한)를 구분해서 저장한다 - truncated는 게임이 끝난 게
        # 아니라 인위적으로 자른 것뿐이라 부트스트랩 방식이 다르다 (compute_returns 참고).
        self.obs_buf = torch.zeros((num_steps, num_envs, *obs_shape), device=self.device)
        self.actions_buf = torch.zeros((num_steps, num_envs), dtype=torch.long, device=self.device)
        self.rewards_buf = torch.zeros((num_steps, num_envs), device=self.device)
        self.values_buf = torch.zeros((num_steps, num_envs), device=self.device)
        self.terminated_buf = torch.zeros((num_steps, num_envs), dtype=torch.bool, device=self.device)
        self.truncated_buf = torch.zeros((num_steps, num_envs), dtype=torch.bool, device=self.device)
        self.trunc_bootstrap_values_buf = torch.zeros((num_steps, num_envs), device=self.device)

        # Track learning progress
        self.training_error = []
        self.q_values = []

    def get_action(self, obs) -> int:
        """단일 관측에 대한 epsilon-greedy 행동 (evaluate()/GIF 녹화용)."""
        if np.random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, device=self.device).unsqueeze(0)
            return int(self.model(obs_t).argmax(dim=1).item())

    def act(self, step, obs, epsilon):
        """rollout 수집 중 num_envs개 관측에 대해 epsilon-greedy로 행동을 고르고,
        obs/value를 이번 step의 rollout buffer에 기록한 뒤 행동을 반환한다."""
        self.obs_buf[step] = obs
        with torch.no_grad():
            values, greedy_actions = self.model(obs).max(dim=1)
        self.values_buf[step] = values

        explore = torch.rand(self.num_envs, device=self.device) < epsilon  # (128, ) bool
        random_actions = torch.randint(0, self.n_actions, (self.num_envs,), device=self.device)  # (128, )
        actions = torch.where(explore, random_actions, greedy_actions)  # explore면 random 아니면 greedy
        self.actions_buf[step] = actions
        return actions

    def record_outcome(self, step, env_index, reward, terminated, truncated):
        """env_index번째 환경이 이번 step에서 받은 결과를 rollout buffer에 기록한다."""
        self.rewards_buf[step, env_index] = reward
        self.terminated_buf[step, env_index] = terminated
        self.truncated_buf[step, env_index] = truncated

    def bootstrap_truncated(self, step, indices, final_obs_batch):
        # MAX_TIME_STEPS으로 truncate된 env들의 reset 전 마지막 observation에서 부트스트랩 값을 미리 계산
        if not indices:
            return
        with torch.no_grad():
            values = self.model(final_obs_batch).max(dim=1).values
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        self.trunc_bootstrap_values_buf[step, index_tensor] = values

    def compute_returns(self, next_obs):
        """모은 rollout 전체에 대해 Q(lambda) return을 뒤에서부터 재귀적으로
        계산한다. target network 없이, 이번 rollout이 끝날 때까지 얼려둔 model의
        값 추정을 그대로 쓰는 셈이라 이게 target network의 대체 역할을 한다.

        terminated/truncated를 구분해서 부트스트랩한다:
          - terminated: 미래 가치 없음 -> return = reward만.
          - truncated: 게임 자체는 계속될 수 있었던 것을 인위적으로 자른 것뿐이라,
            reset 전 마지막 observation에서 미리 계산해둔 값으로 1-step만
            부트스트랩한다 (다음 episode로 넘어간 return/value와는 연결하지 않는다
            - 그러면 episode 경계를 넘게 된다).
          - 그 외(episode가 계속됨): 표준 Q(lambda) 재귀.

        Q(lambda)
        G_t = r_t + gamma * (lambda * G_{t+1} + ((1-lambda) * V(s_{t+1}))

        t+1 시점이 필요하므로 뒤부터 앞으로 계산해야함
        """
        with torch.no_grad():
            returns = torch.zeros_like(self.rewards_buf)

            for t in reversed(range(self.num_steps)):
                reward_t = self.rewards_buf[t]
                terminated_t = self.terminated_buf[t]
                truncated_t = self.truncated_buf[t]

                if t == self.num_steps - 1:
                    # normal q return 
                    normal_return = reward_t + self.gamma * self.model(next_obs).max(dim=1).values
                else:
                    # q lambda return
                    lambda_bootstrap = (
                        self.q_lambda * returns[t + 1]
                        + (1.0 - self.q_lambda) * self.values_buf[t + 1]
                    )
                    normal_return = reward_t + self.gamma * lambda_bootstrap

                truncated_return = reward_t + self.gamma * self.trunc_bootstrap_values_buf[t]

                # truncated와 terminated 된 부분 처리
                target = torch.where(truncated_t, truncated_return, normal_return)
                target = torch.where(terminated_t, reward_t, target)
                returns[t] = target
        return returns

    def learn(self, returns, num_minibatches, update_epochs, data_augmentation=False, augmentation_mode="spawn_safe"):
        """
        rollout 과 return으로 update_epochs만큼 학습
        return mean loss, mean q
        """
        obs_shape = self.obs_buf.shape[2:]
        batch_size = self.num_envs * self.num_steps
        minibatch_size = batch_size // num_minibatches

        batch_obs = self.obs_buf.reshape(batch_size, *obs_shape)
        batch_actions = self.actions_buf.reshape(batch_size, 1)
        batch_returns = returns.reshape(batch_size)

        indices = np.arange(batch_size)
        iteration_losses = []
        iteration_q_values = []
        for _ in range(update_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, minibatch_size):
                minibatch = indices[start:start + minibatch_size]
                mb_obs = batch_obs[minibatch]
                mb_actions = batch_actions[minibatch]
                if data_augmentation:
                    # return은 이미 계산되어 있으므로 (obs, actions)만 변환하면됨
                    mb_obs, mb_actions, _ = augment_transitions(
                        mb_obs, mb_actions, mb_obs, augmentation_mode
                    )
                predicted = self.model(mb_obs).gather(1, mb_actions).squeeze(1)

                if self.loss_fn == "huber":
                    loss = F.smooth_l1_loss(batch_returns[minibatch], predicted)
                else:
                    loss = F.mse_loss(batch_returns[minibatch], predicted)

                self.optimizer.zero_grad()
                loss.backward()
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                self.training_error.append(loss.item())
                self.q_values.append(predicted.mean().item())
                iteration_losses.append(loss.item())
                iteration_q_values.append(predicted.mean().item())

        return float(np.mean(iteration_losses)), float(np.mean(iteration_q_values))

    def sample_features(self, sample_size):
        """srank(표현 rank) 계산용으로, 이번 rollout에서 무작위로 sample_size개
        관측을 뽑아 penultimate feature를 반환한다."""
        obs_shape = self.obs_buf.shape[2:]
        batch_size = self.num_envs * self.num_steps
        batch_obs = self.obs_buf.reshape(batch_size, *obs_shape)
        sample_size = min(sample_size, batch_size)
        sample_idx = torch.from_numpy(np.random.choice(batch_size, sample_size, replace=False))
        with torch.no_grad():
            return self.model.forward_features(batch_obs[sample_idx])

    def set_learning_rate(self, lr):
        for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = lr


def evaluate(agent, env, num_episodes=20, seed=None, epsilon=0.0):
    """탐색/학습 없이 정책을 평가한다.

    Danmaku의 성능 지표는 reward 합이 아니라 game.state.score (= 생존 초 수)다.
    DanmakuImgEnv가 MAX_TIME_STEPS(10800 물리 스텝, score 180)에서 스스로 truncate
    하므로 여기서 인위적인 스텝 상한을 두지 않는다. 상한을 두면 목표 점수 120을
    측정할 수 없게 된다.

    random policy if agent == None
    """
    scores = []
    lengths = []
    returns = []
    action_counts = np.zeros(env.n_actions, dtype=np.int64)

    # agent를 None으로 두면, random agent로 작동
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
        "sem_score": (  # 평균 점수의 표준 오차 (standard error of the mean)
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

