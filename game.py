import config
from random import randint
from entity import Ball, Agent 
from dataclasses import dataclass, field 


@dataclass 
class GameState: 
    agent: Agent = None
    balls: list = field(default_factory=list)
    steps: int = 0
    alive: bool = True 
    score: int = 0 

def _spawn_balls(balls):
    if len(balls) < config.MAX_BALL_NUM: 
        balls.append(Ball(0, 0 , 
                          vx = randint(config.MIN_BALL_SPEED, config.MAX_BALL_SPEED),
                          vy = randint(config.MIN_BALL_SPEED, config.MAX_BALL_SPEED)))

        
def _is_Collision(agent, ball): 
    dx = agent.x - ball.x
    dy = agent.y - ball.y
    r = agent.r + ball.r
    return dx ** 2 + dy ** 2 <= r ** 2 
 
def _reflect(ball): 
    if ball.x - ball.r < 0: 
        ball.x = ball.r 
        ball.vx = -ball.vx
    elif ball.x + ball.r > config.SCREEN_WIDTH: 
        ball.x = config.SCREEN_WIDTH - ball.r 
        ball.vx = -ball.vx 

    if ball.y - ball.r < 0: 
        ball.y = ball.r
        ball.vy = -ball.vy 
    elif ball.y + ball.r > config.SCREEN_HEIGHT: 
        ball.y = config.SCREEN_HEIGHT - ball.r 
        ball.vy = -ball.vy 

def _move_agent(agent, action): 
    if action ==  config.ACTION_UP: 
        agent.y -= agent.speed 
    elif action == config.ACTION_DOWN:
        agent.y += agent.speed 
    elif action == config.ACTION_LEFT: 
        agent.x -= agent.speed
    elif action == config.ACTION_RIGHT: 
        agent.x += agent.speed
    elif action == config.ACTION_UP_LEFT: 
        agent.y -= config.AGENT_DIAG_SPEED
        agent.x -= config.AGENT_DIAG_SPEED
    elif action == config.ACTION_UP_RIGHT: 
        agent.y -= config.AGENT_DIAG_SPEED
        agent.x += config.AGENT_DIAG_SPEED
    elif action == config.ACTION_DOWN_LEFT: 
        agent.y += config.AGENT_DIAG_SPEED
        agent.x -= config.AGENT_DIAG_SPEED
    elif action == config.ACTION_DOWN_RIGHT: 
        agent.y += config.AGENT_DIAG_SPEED
        agent.x += config.AGENT_DIAG_SPEED
    agent.x = max(agent.r, min(config.SCREEN_WIDTH - agent.r, agent.x))
    agent.y = max(agent.r, min(config.SCREEN_HEIGHT - agent.r, agent.y))
    


class Game: 
    def __init__(self): 
        self.state = GameState()
        self.reset()
    def reset(self):
        state = self.state 
        state.agent = Agent(x = config.SCREEN_WIDTH//2, y = config.SCREEN_HEIGHT//2)
        state.balls = []
        state.steps = 0 
        state.alive = True 
        state.score = 0 
    def step(self, action): 
        state = self.state
        if not state.alive: return 
        _move_agent(state.agent, action)

        for ball in state.balls: 
            ball.x += ball.vx
            ball.y += ball.vy
            if _is_Collision(state.agent, ball): 
                print(state.score)
                state.alive = False 
                return 
            _reflect(ball)

        if state.steps % config.SCORE_INTERVAL == 0: 
            state.score += 1

        if state.steps % config.BALL_ADD_FREQUENCY == 0: 
            _spawn_balls(state.balls)
        state.steps += 1 
