from game import Game
from render import Renderer
import config 
import pygame 

import numpy as np
from collections import deque
from PIL import Image
import random 

from gymnasium import Env
from gymnasium.spaces import Box, Discrete

class DanmakuEnv(Env): 
    def __init__(self): 
        self.game = Game()
        self.action_space = Discrete(9)
        self.image_size = (84, 84)
        self.stack_size = config.N_FRAME_STACK
        self.max_time_steps = config.MAX_TIME_STEPS
        self.observation_space = Box(low = 0, high = 255, shape = (4, 84, 84), dtype = np.int8)
        self.frames = deque(maxlen = config.N_FRAME_STACK)
        self.renderer = Renderer()

    def reset(self, seed = None):
        super().reset(seed=seed)

        if seed is not None: 
            random.seed(seed)

        self.game.reset()
        self.frames.clear()
        first_obs = self._get_obs()

        for _ in range(self.stack_size):
            self.frames.append(first_obs) 

        observation = np.stack(self.frames, axis = 0)
        info = self._get_info()
        return observation, info 

    def step(self, action):
        total_reward= 0
        terminated = False
        truncated = False 

        for _ in range(config.N_FRAME_SKIP): 
            self.game.step(action)
            terminated = not self.game.state.alive
            truncated = self.game.state.steps >= self.max_time_steps
            reward = 0 if terminated else 1
            total_reward += reward 

            if terminated or truncated: 
                break

        curr_obs = self._get_obs() 
        observation = self._frame_stack(curr_obs)
        info = self._get_info() # info return 
        return observation, total_reward, terminated, truncated, info
    
    def _frame_stack(self, obs):
        self.frames.append(obs)
        return np.stack(self.frames, axis = 0)
        
    def _get_obs(self): # 현재 state -> RGB렌더 =-> image_to_gray -> resize_image -> return (84,84)
        self.renderer.draw(self.game, view_score = False)
        rgb_array = pygame.surfarray.array3d(self.renderer.screen)
        rgb_array = rgb_array.transpose(1,0,2)

        gray_img = self.image_to_gray(rgb_array)
        resize_img = self.resize_image(gray_img)
        return np.array(resize_img, dtype=np.int8)

    def image_to_gray(self, arr):
        return Image.fromarray(arr, mode="RGB").convert("L")

    def resize_image(self, img): 
        return img.resize(self.image_size)
    
    def _get_info(self):
        return {}
