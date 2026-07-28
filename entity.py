import config 
from dataclasses import dataclass
import random 


@dataclass
class Agent:
    x: float
    y: float
    radius: int = config.AGENT_RADIUS
    speed: int = config.AGENT_SPEED
    


@dataclass
class Ball:
    #def __init__(self): 
    x: float
    y: float
    r: int = config.BALL_RADIUS
    vx: int = random.randint(config.MIN_BALL_SPEED, config.MAX_BALL_SPEED)
    vy: int = random.randint(config.MIN_BALL_SPEED, config.MAX_BALL_SPEED)
    




