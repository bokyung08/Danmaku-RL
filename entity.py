import config
import random

class Agent:
    x: float
    y: float
    r: int
    speed: float

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.r = config.AGENT_RADIUS
        self.speed = config.AGENT_SPEED


class Ball:
    x: float
    y: float
    vx: float
    vy: float
    r: int

    def __init__(self, x, y, vx, vy):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.r = config.BALL_RADIUS