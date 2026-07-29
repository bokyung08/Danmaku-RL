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
        balls.append(Ball(x = config.BALL_RADIUS,
                          y = config.BALL_RADIUS, 
                          vx = randint(config.MIN_BALL_SPEED, config.MAX_BALL_SPEED),
                          vy = randint(config.MIN_BALL_SPEED, config.MAX_BALL_SPEED)))

# the simple version of collision
# def _is_collision(agent, ball): 
#     dx = agent.x - ball.x
#     dy = agent.y - ball.y
#     r = agent.r + ball.r
#     return dx ** 2 + dy ** 2 <= r ** 2 

# frame까지 고려한 collision
def _is_collision(agent_prev, agent, ball_prev, ball):
    prev_agent_x, prev_agent_y = agent_prev
    prev_ball_x, prev_ball_y = ball_prev

    px, py = prev_ball_x - prev_agent_x, prev_ball_y - prev_agent_y
    agent_vx, agent_vy = agent.x - prev_agent_x, agent.y - prev_agent_y
    ball_vx, ball_vy = ball.x - prev_ball_x, ball.y - prev_ball_y

    R = agent.r + ball.r

    # 상대속도
    vx, vy = ball_vx - agent_vx, ball_vy - agent_vy

    a = vx**2 + vy**2
    b = 2 * (px * vx + py * vy)
    c = px**2 + py**2 - R**2

    if c <= 0: return True  # frame 시작시 원 내부인가? 
    if a == 0: return False  # 상대 위치가 그대로임. 


    discriminant = b**2 - 4*a*c

    if discriminant < 0: return False
    else:
        hit_t = (-b - discriminant**(1/2)) / (2*a)
        if 0 <= hit_t <= 1: return True
        else: return False



def _reflect(ball): 
    if ball.x - ball.r < 0: 
        ball.x = 2 * ball.r - ball.x
        ball.vx = -ball.vx
    elif ball.x + ball.r > config.SCREEN_WIDTH: 
        ball.x = 2 * config.SCREEN_WIDTH - 2 * ball.r - ball.x
        ball.vx = -ball.vx 

    if ball.y - ball.r < 0: 
        ball.y = 2 * ball.r - ball.y
        ball.vy = -ball.vy 
    elif ball.y + ball.r > config.SCREEN_HEIGHT: 
        ball.y = 2 * config.SCREEN_HEIGHT - 2 * ball.r - ball.y
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
        if 2 * (config.AGENT_RADIUS+config.BALL_RADIUS) < config.AGENT_SPEED + config.MAX_BALL_SPEED:
            print("With this setting, the ball can pass through the agent. Reduce the speed or size of the ball and the agent.")
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

        agent_prev = (state.agent.x, state.agent.y)
        _move_agent(state.agent, action)

        for ball in state.balls: 
            ball_prev = (ball.x, ball.y)
            ball.x += ball.vx
            ball.y += ball.vy

            if _is_collision(agent_prev, state.agent, ball_prev, ball): 
                print(state.score)
                state.alive = False 
                return 
            _reflect(ball)

        if state.steps % config.BALL_ADD_FREQUENCY == 0:
                    _spawn_balls(state.balls)
                    
        state.steps += 1

        if state.steps % config.SCORE_INTERVAL == 0:
            state.score += 1

        
        #print(len(state.balls))
