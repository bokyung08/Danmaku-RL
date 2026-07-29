from game import Game
from entity import Ball, Agent 
import config 
from render import Renderer

import numpy as np
from collections import deque
from PIL import Image
import random 
import gymnasium as gym
from gymnasium.spaces import Box, Discrete

class DanmakuEnv(gym.Env): 
    def __init__(self): 
        self.game = Game()
        self.action_space = Discrete(9)
        self.image_size = (84, 84)
        self.stack_size = config.N_FRAME_STACK
        self.max_time_steps = config.MAX_TIME_STEPS
        self.observation_space = Box(
            low=0, high=255, shape=(self.stack_size, *self.image_size), dtype=np.uint8
        )
        self.frames = deque(maxlen = config.N_FRAME_STACK)
        self.renderer = Renderer(render_mode="rgb_array")

    def reset(self, seed = None, options=None):
        super().reset(seed=seed)

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
        reward= 0
        terminated = False
        truncated = False 
        prev_score = self.game.state.score
        for _ in range(config.N_FRAME_SKIP): 
            self.game.step(action)
            terminated = not self.game.state.alive
            truncated = self.game.state.steps >= self.max_time_steps
            
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
        return Image.fromarray(arr).convert("L")

    def resize_image(self, image): 
        resized = image.resize(
            self.image_size,
            Image.Resampling.BILINEAR,
        )
        return np.asarray(resized, dtype=np.uint8)


    def _frame_stack(self, obs):
        self.frames.append(obs)
        return np.stack(self.frames, axis = 0)

