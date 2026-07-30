from game import Game
import config 
from render import Renderer

import numpy as np
from collections import deque
from PIL import Image
import random 

class DanmakuEnv: 
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
        if seed is not None: 
            random.seed(seed)

        self.game.reset()
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

