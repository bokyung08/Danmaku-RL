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

class DanmakuVecEnv(gym.Env): 
    def __init__(self): 
        self.game = Game()
        self.action_space = Discrete(9)
        
        self.max_time_steps = config.MAX_TIME_STEPS
        self.observation_space = Box(
            low=0, high=255, shape=(self.stack_size, *self.image_size), dtype=np.uint8
        )
    
        self.renderer = Renderer(render_mode="rgb_array")

    def reset(self, seed = None, options=None):
        super().reset(seed=seed)

        if seed is not None: 
            random.seed(seed)

        self.game.reset()
        self.frames.clear()
        observation = self._get_obs()
        info = self._get_info()
        return observation, info 

    def step(self, action):
        reward= 0
        terminated = False
        truncated = False 
        prev_score = self.game.state.score

        self.game.step(action)
        terminated = not self.game.state.alive
        truncated = self.game.state.steps >= self.max_time_steps
                    
        if terminated or truncated: 
            return 
        reward = self.game.state.score - prev_score
        observation =self._get_obs() 
        info = self._get_info() # info return 
        
        return observation, reward, terminated, truncated, info
        
    def _get_obs(self): # 현재 state -> RGB렌더 =-> image_to_gray -> resize_image -> return (84,84)
        pass
