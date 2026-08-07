from game import Game
import config
from render import Renderer

import numpy as np
from collections import deque


class DanmakuImgEnv:
    def __init__(self):
        self.game = Game()
        self.action_space = range(9)
        self.n_actions = len(self.action_space)

        self.image_size = (84, 84)
        self.stack_size = config.N_FRAME_STACK
        self.observation_shape = (self.stack_size, *self.image_size)

        self.max_time_steps = config.MAX_TIME_STEPS
        self.frames = deque(maxlen = config.N_FRAME_STACK)
        self.renderer = Renderer(render_mode="rgb_array")

    def reset(self, seed = None):
        self.game.reset(seed=seed)
        self.frames.clear()
        first_obs = self._get_obs()

        for _ in range(self.stack_size):
            self.frames.append(first_obs) #stack frame 처음 네 프레임 쌓기

        observation = np.stack(self.frames, axis = 0)  # (4,84,84)
        info = self._get_info()
        return observation, info 

    def step(self, action):
        terminated = False
        truncated = False 
        for _ in range(config.N_FRAME_SKIP): 
            self.game.step(action)
            terminated = not self.game.state.alive
            truncated = not terminated and self.game.state.steps >= self.max_time_steps
            
            if terminated or truncated: 
                break
        reward = 0.01 if not terminated else -1.0
        curr_obs = self._get_obs() 
        observation = self._frame_stack(curr_obs)
        info = self._get_info() # info return 
        if truncated:
            info["final_observation"] = observation.copy()
        
        return observation, reward, terminated, truncated, info
        
    def _get_obs(self): # 현재 state -> 흑백 렌더 -> (84,84)로 축소해서 return
        return self.renderer.get_grayscale_image(self.game, size=self.image_size)

    def _get_info(self):
        return {
            "score": self.game.state.score,
            "steps": self.game.state.steps
        }


    def _frame_stack(self, obs):
        self.frames.append(obs)
        return np.stack(self.frames, axis = 0)

    # 환경 종료시 renderer 종료
    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

class DanmakuVecEnv: 
    AGENT_FEATURE_NUM = 2
    BALL_FEATURE_NUM = 5

    def __init__(self, normalize="none"):
        if normalize not in ("none", "near"):
            raise ValueError("normalize must be 'none' or 'near'")
        self.normalize = normalize
        self.game = Game()
        self.action_space = range(9)
        self.observation_shape = (
            self.AGENT_FEATURE_NUM + self.BALL_FEATURE_NUM * config.MAX_BALL_NUM,
        )
        self.n_actions = len(self.action_space)
        self.max_time_steps = config.MAX_TIME_STEPS
        
    def reset(self, seed = None):
        self.game.reset(seed=seed)
        observation = self._get_obs()
        info = self._get_info()
        return observation, info 

    def step(self, action):
        terminated = False
        truncated = False 

        for _ in range(config.N_FRAME_SKIP):
            self.game.step(action)
            terminated = not self.game.state.alive
            truncated = not terminated and (self.game.state.steps >= self.max_time_steps)
            if terminated or truncated: break

        reward = 0.01 if not terminated else -1.0

        observation = self._get_obs() 
        info = self._get_info() # info return 
        if truncated:
            info["final_observation"] = observation.copy()
        
        return observation, reward, terminated, truncated, info

    # 예전 시도: ball이 agent와 충돌할 때까지 걸리는 시간을 계산하여 가까운 순으로 정렬
    # @staticmethod
    # def _time_to_collision(agent, ball):
    #     px = ball.x - agent.x
    #     py = ball.y - agent.y
    #     vx = ball.vx
    #     vy = ball.vy
    #     collision_radius = agent.r + ball.r
    #
    #     a = vx**2 + vy**2
    #     b = 2.0 * (px * vx + py * vy)
    #     c = px**2 + py**2 - collision_radius**2
    #
    #     if c <= 0.0:
    #         return 0.0
    #     if a == 0.0:
    #         return float("inf")
    #
    #     discriminant = b**2 - 4.0 * a * c
    #     if discriminant < 0.0:
    #         return float("inf")
    #
    #     hit_time = (-b - discriminant**0.5) / (2.0 * a)
    #     return hit_time if hit_time >= 0.0 else float("inf")
    #
    # def _get_obs(self):
    #     state = self.game.state
    #     agent = state.agent
    #     obs = np.zeros(self.AGENT_FEATURE_NUM + self.BALL_FEATURE_NUM * config.MAX_BALL_NUM, dtype=np.float32)
    #
    #     min_agent_x = agent.r
    #     max_agent_x = config.SCREEN_WIDTH - agent.r
    #     min_agent_y = agent.r
    #     max_agent_y = config.SCREEN_HEIGHT - agent.r
    #     obs[0] = (2.0 * (agent.x - min_agent_x)/(max_agent_x - min_agent_x) - 1.0)
    #     obs[1] = (2.0 * (agent.y - min_agent_y)/(max_agent_y - min_agent_y) - 1.0)
    #
    #     sorted_balls = sorted(
    #         state.balls,
    #         key=lambda ball: (
    #             self._time_to_collision(agent, ball),
    #             (ball.x - agent.x) ** 2 + (ball.y - agent.y) ** 2,
    #         ),
    #     )
    #     for idx, ball in enumerate(sorted_balls[:config.MAX_BALL_NUM]):
    #         start = self.AGENT_FEATURE_NUM + idx * self.BALL_FEATURE_NUM
    #         relative_x = (ball.x - agent.x) / config.SCREEN_WIDTH
    #         relative_y = (ball.y - agent.y) / config.SCREEN_HEIGHT
    #         normalized_vx = ball.vx / config.MAX_BALL_SPEED
    #         normalized_vy = ball.vy / config.MAX_BALL_SPEED
    #         remaining_time = self._time_to_collision(agent, ball)
    #         collision_urgency = 0.0 if np.isinf(remaining_time) else (
    #             config.PHYSICS_FPS / (remaining_time + config.PHYSICS_FPS)
    #         )
    #         obs[start:start+self.BALL_FEATURE_NUM] = [
    #             relative_x, relative_y, normalized_vx, normalized_vy, collision_urgency,
    #         ]
    #     return np.clip(obs, -1.0, 1.0)

    def _get_obs(self, normalize=None):
        """
        GameState를 고정된 1차원 벡터로 변환한다. ``near``는 공을 agent와
        가까운 순서로 정렬하는 예전 MLP 관측이고, ``none``은 attention용으로
        spawn 순서를 유지한다.

        [a(x), a(y), b1(x), b1(y), b1(vx), b1(vy), b1(mask), b2(x), ...]
        agent/ball의 (x, y) 모두 화면 기준 절대 위치를 [-1, 1]로 정규화한다.
        mask는 1.0이면 실제 공, 0.0이면 빈 슬롯(패딩)을 뜻한다.
        """

        normalize = self.normalize if normalize is None else normalize
        if normalize not in ("none", "near"):
            raise ValueError("normalize must be 'none' or 'near'")

        state = self.game.state
        agent = state.agent
        obs_size = (self.AGENT_FEATURE_NUM + self.BALL_FEATURE_NUM * config.MAX_BALL_NUM)
        obs = np.zeros(obs_size, dtype=np.float32)

        min_agent_x = agent.r # 에이전트의 반지름 만큼 떨어져 있는 곳이 min 좌표
        max_agent_x = config.SCREEN_WIDTH - agent.r
        min_agent_y = agent.r
        max_agent_y = config.SCREEN_HEIGHT - agent.r

        obs[0] = (2.0 * (agent.x - min_agent_x)/(max_agent_x - min_agent_x) - 1.0)
        obs[1] = (2.0 * (agent.y - min_agent_y)/(max_agent_y - min_agent_y) - 1.0)

        balls = state.balls
        if normalize == "near":
            balls = sorted(
                balls,
                key=lambda ball: (
                    (ball.x - agent.x) ** 2 + (ball.y - agent.y) ** 2
                ),
            )
        for idx, ball in enumerate(balls[:config.MAX_BALL_NUM]):
            start = self.AGENT_FEATURE_NUM + idx * self.BALL_FEATURE_NUM

            min_ball_x = ball.r # 공의 반지름 만큼 떨어져 있는 곳이 min 좌표
            max_ball_x = config.SCREEN_WIDTH - ball.r
            min_ball_y = ball.r
            max_ball_y = config.SCREEN_HEIGHT - ball.r

            normalized_x = (2.0 * (ball.x - min_ball_x)/(max_ball_x - min_ball_x) - 1.0)
            normalized_y = (2.0 * (ball.y - min_ball_y)/(max_ball_y - min_ball_y) - 1.0)
            normalized_vx = ball.vx / config.MAX_BALL_SPEED
            normalized_vy = ball.vy / config.MAX_BALL_SPEED

            obs[start:start+self.BALL_FEATURE_NUM] = [
                normalized_x,
                normalized_y,
                normalized_vx,
                normalized_vy,
                1.0,
            ]

        return np.clip(obs, -1.0, 1.0)

    def _get_info(self):
        return {
            "score": self.game.state.score,
            "steps": self.game.state.steps,
        }

    def close(self):
        pass
