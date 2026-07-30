import config
import random
import math

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

    def __init__(self):
        # 첫 충돌 판정을 무시하기 위해 반지름만큼 떨어진 거리에 공 생성
        self.x = config.BALL_RADIUS+1
        self.y = config.BALL_RADIUS+1

        # v in range(1,16), theta ~ Unif(3/2 * pi, 2 pi)
        self.v = random.uniform(config.MIN_BALL_SPEED, config.MAX_BALL_SPEED)
        self.theta = random.uniform(0.05 * math.pi, 0.45 * math.pi)  # 벽면과 너무 평행하지 않게 띄워줌

        # vx = v * cos(theta), vy = v * sin(theta)
        self.vx = self.v * math.cos(self.theta)
        self.vy = self.v * math.sin(self.theta)
        self.r = config.BALL_RADIUS