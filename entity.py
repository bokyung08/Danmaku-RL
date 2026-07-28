import config
import random

class Agent:
    x: int
    y: int
    r: int
    speed: int

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.r = config.AGENT_RADIUS
        self.speed = config.AGENT_SPEED


class Ball:
    x: int
    y: int
    vx: int
    vy: int
    r: int

    def __init__(self):
        # 첫 충돌 판정을 무시하기 위해 반지름만큼 떨어진 거리에 공 생성
        self.x = config.BALL_RADIUS
        self.y = config.BALL_RADIUS

        self.vx = random.randint(config.MIN_BALL_SPEED, config.MAX_BALL_SPEED)
        self.vy = random.randint(config.MIN_BALL_SPEED, config.MAX_BALL_SPEED)
        self.r = config.BALL_RADIUS