from game import Game
import config 
from render import Renderer

import numpy as np
from collections import deque
from PIL import Image

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
        prev_score = self.game.state.score
        for _ in range(config.N_FRAME_SKIP): 
            self.game.step(action)
            terminated = not self.game.state.alive
            truncated = not terminated and self.game.state.steps >= self.max_time_steps
            
            if terminated or truncated: 
                break
        reward = self.game.state.score - prev_score
        curr_obs = self._get_obs() 
        observation = self._frame_stack(curr_obs)
        info = self._get_info() # info return 
        
        return observation, reward, terminated, truncated, info
        
    def _get_obs(self): # 현재 state -> RGB렌더 =-> image_to_gray -> resize_image -> return (84,84)
        # 현재 게임 state를 render
        image = self.renderer.get_image(self.game)  # (600,600,3)
        grayscale_img = self.image_to_gray(image)  # (600,600)
        resize_img = self.resize_image(grayscale_img)  # (84, 84)
        return resize_img
    
    def _get_info(self):
        return {
            "score": self.game.state.score,
            "steps": self.game.state.steps
        }

    def image_to_gray(self, arr):
        # L: 각 pixel을 밝기값 하나로 표현
        return Image.fromarray(arr).convert("L")  # 다른 grayscale을 적용해 볼 수도 있음

    def resize_image(self, image): 
        # 주변 pixel을 선형 보간하여 새 pixel 값 계산 (부드러워짐)
        resized = image.resize(
            self.image_size,
            Image.Resampling.BILINEAR,  # NEAREST, BILINEAR, HAMMING, BICUBIC, LANCZOS, 또는 maxpool도 써볼 수 있음
        )
        return np.asarray(resized, dtype=np.uint8)


    def _frame_stack(self, obs):
        self.frames.append(obs)
        return np.stack(self.frames, axis = 0)

    # 환경 종료시 renderer 종료
    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

class DanmakuVecEnv: 
    AGENT_FEATURE_NUM = 4
    BALL_FEATURE_NUM = 5

    def __init__(self): 
        self.game = Game()
        self.action_space = range(9)
        self.observation_shape = (self.AGENT_FEATURE_NUM + self.BALL_FEATURE_NUM * config.MAX_BALL_NUM)
        self.max_time_steps = config.MAX_TIME_STEPS
        
    def reset(self, seed = None):
        self.game.reset(seed=seed)
        observation = self._get_obs()
        info = self._get_info()
        return observation, info 

    def step(self, action):
        reward= 0
        terminated = False
        truncated = False 

        agent = self.game.state.agent
        prev_agent_x = agent.x
        prev_agent_y = agent.y 
        prev_score = self.game.state.score

        self.game.step(action)
        terminated = not self.game.state.alive
        truncated = not terminated and (self.game.state.steps >= self.max_time_steps)
                    
        agent = self.game.state.agent
        self.agent_vx = (agent.x - prev_agent_x) 
        self.agent_vy = (agent.y - prev_agent_y)

        reward = self.game.state.score - prev_score

        observation = self._get_obs() 
        info = self._get_info() # info return 
        
        return observation, reward, terminated, truncated, info
        
    def _get_obs(self): 
        '''
        GameState를 고정된 1차원 벡터로 변환한다.
        Balls -> ball에서 x, y, vx, vy를 뽑아옴 
        agent의 x,y와 각 ball의 x, y간의 거리 계산을 통해 distances[d1,d2,d3] 계산 후 오름차순 sort 
        [a(x), a(y), a(vx), a(vy), b1(x), b1(y), b1(vx), b1(vy), b1(mask = 1), b2(x), b2(y), b2(vx), b2(vy), b2(mask = 1),0,0,0 ...]        

        obs에 들어가는 agent와 ball의 x, y 값은 raw값이 아닌 [-1, 1] 로  정규화 된 값이다. 

        ''' 
        
        state = self.game.state
        agent = state.agent 
        obs_size = (self.AGENT_FEATURE_NUM + self.BALL_FEATURE_NUM * config.MAX_BALL_NUM)
        obs = np.zeros(obs_size, dtype=np.float32) # 공, 에이전트 없는 상태 (패딩 주기 위해 zeros로 초기화)

        min_agent_x = agent.r # 에이전트의 반지름 만큼 떨어져 있는 곳이 min 좌표 
        max_agent_x = config.SCREEN_WIDTH - agent.r 
        min_agent_y = agent.r
        max_agent_y = config.SCREEN_HEIGHT - agent.r 

        obs[0] = (2.0 * (agent.x - min_agent_x)/(max_agent_x - min_agent_x) - 1.0) 
        obs[1] = (2.0 * (agent.y - min_agent_y)/(max_agent_y - min_agent_y) - 1.0)
        obs[2] = self.agent_vx / config.AGENT_SPEED # 에이전트 1스텝 이동량 = [-5, 5] 사이 -> 관측공간 크기에 맞추기 위해 SPEED로 나눔 
        obs[3] = self.agent_vy / config.AGENT_SPEED

        sorted_balls = sorted(state.balls, key = lambda ball: ((ball.x - agent.x) ** 2 + (ball.y - agent.y) ** 2), ) # 거리 순 정규화 
        for idx, ball in enumerate(sorted_balls[:config.MAX_BALL_NUM]):  
            start = self.AGENT_FEATURE_NUM + idx * self.BALL_FEATURE_NUM 

            min_ball_x = ball.r # 공의 반지름 만큼 떨어져 있는 곳이 min 좌표 
            max_ball_x = config.SCREEN_WIDTH - ball.r 
            min_ball_y = ball.r
            max_ball_y = config.SCREEN_HEIGHT - ball.r 
            
            normalized_x = (2.0 * (ball.x - min_ball_x)/(max_ball_x - min_ball_x) - 1.0) 
            normalized_y = (2.0 * (ball.y - min_ball_y)/(max_ball_y - min_ball_y) - 1.0)

            # 공의 상대 좌표 (실험을 위해 남겨둠)
            # relative_x = (ball.x - agent.x) / config.SCREEN_WIDTH 
            # relative_y = (ball.y - agent.y) / config.SCREEN_HEIGHT

            normalized_vx = ball.vx / config.MAX_BALL_SPEED
            normalized_vy = ball.vy / config.MAX_BALL_SPEED

            obs[start:start+self.BALL_FEATURE_NUM] = [normalized_x, normalized_y, normalized_vx, normalized_vy, 1.0]

        return np.clip(obs, -1.0, 1.0)

    def _get_info(self):
        return {
            "score": self.game.state.score,
            "steps": self.game.state.steps,
        }