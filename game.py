"""순수 게임 로직. pygame 을 절대 import 하지 않는다."""
from dataclasses import dataclass, field

import config
from entity import Ball, Agent


@dataclass
class GameState:
    agent: Agent = None
    balls: list = field(default_factory=list)
    steps: int = 0
    alive: bool = True
    

def _spawn_balls(balls):
    if len(balls)<=config.MAX_BALL_NUM: 
        balls.append(Ball(0,0))


def _move_agent(agent, action):
    if action == config.ACTION_UP:
        agent.y -= agent.speed
    elif action == config.ACTION_DOWN:
        agent.y += agent.speed
    elif action == config.ACTION_LEFT:
        agent.x -= agent.speed
    elif action == config.ACTION_RIGHT:
        agent.x += agent.speed
    elif action == config.ACTION_UP_LEFT:
        agent.x -= config.AGENT_DIAG_SPEED
        agent.y -= config.AGENT_DIAG_SPEED
    elif action == config.ACTION_UP_RIGHT:
        agent.x += config.AGENT_DIAG_SPEED
        agent.y -= config.AGENT_DIAG_SPEED
    elif action == config.ACTION_DOWN_LEFT:
        agent.x -= config.AGENT_DIAG_SPEED
        agent.y += config.AGENT_DIAG_SPEED
    elif action == config.ACTION_DOWN_RIGHT:
        agent.x += config.AGENT_DIAG_SPEED
        agent.y += config.AGENT_DIAG_SPEED
        
    agent.x = max(agent.radius, min(config.SCREEN_WIDTH - agent.radius, agent.x))
    agent.y = max(agent.radius, min(config.SCREEN_HEIGHT - agent.radius, agent.y))



def _reflect(ball):
    if ball.x - ball.r < 0:
        ball.x = ball.r
        ball.vx = -ball.vx
    elif ball.x + ball.r > config.SCREEN_WIDTH:
        ball.x = config.SCREEN_WIDTH - ball.radius
        ball.vx = -ball.vx

    if ball.y - ball.r < 0:
        ball.y = ball.r
        ball.vy = -ball.vy
    elif ball.y + ball.r > config.SCREEN_HEIGHT:
        ball.y = config.SCREEN_HEIGHT - ball.r
        ball.vy = -ball.vy


def _is_collision(agent, ball):
    dx = agent.x - ball.x
    dy = agent.y - ball.y
    r = agent.radius + ball.r
    return dx * dx + dy * dy <= r * r


class Game:
    def __init__(self):
        self.state = GameState()
        self.reset()

    def reset(self):
        state = self.state
        state.agent = Agent(x=config.SCREEN_WIDTH / 2, y=config.SCREEN_HEIGHT / 2)
        state.balls = []
        state.steps = 0
        state.alive = True
    

    def step(self, action):
        state = self.state

        _move_agent(state.agent, action)

        for ball in state.balls:
            ball.x += ball.vx
            ball.y += ball.vy
            if _is_collision(state.agent, ball): 
                print(state.steps)
                return self.reset()
            _reflect(ball)

        if state.steps % config.BALL_ADD_FREQUENCY == 0:
            _spawn_balls(state.balls)
        
        state.steps += 1