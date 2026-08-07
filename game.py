import config
import random
from entity import Ball, Agent 
from dataclasses import dataclass, field 


@dataclass 
class GameState: 
    agent: Agent = None
    # 다른 gamestate 객체가 list를 공유할 위험 있으므로 field 사용
    balls: list = field(default_factory=list)  
    steps: int = 0
    alive: bool = True 
    score: int = 0 

def _spawn_balls(balls, rng):
    if len(balls) < config.MAX_BALL_NUM:
        balls.append(Ball(rng))



def _lerp(p0, p1, t):
    x0, y0 = p0
    x1, y1 = p1
    xt = x0 + t * (x1 - x0)
    yt = y0 + t * (y1 - y0)
    return (xt, yt)


# 공을 한 프레임 움직이고 그동안 벽 반사와 충돌 감지 확인
def _move_ball(ball, agent_prev, agent_next, agent_r):
    elapsed = 0.0  # 흐른 시간 (0~1)

    # 최대 두번까지 반사 고려
    for _ in range(3):
        remain = 1 - elapsed  # 사용 가능한 시간

        if remain <= 0: break

        # tx, ty: 벽과 부딪힐 때까지 남은 시간 
        if ball.vx > 0:
            tx = (config.SCREEN_WIDTH - ball.r - ball.x) / ball.vx
        elif ball.vx < 0:
            tx = (ball.r - ball.x) / ball.vx
        else:
            tx = float('inf')

        if ball.vy > 0:
            ty = (config.SCREEN_HEIGHT - ball.r - ball.y) / ball.vy
        elif ball.vy < 0:
            ty = (ball.r - ball.y) / ball.vy
        else: ty = float('inf')

        dt = min(tx, ty, remain)  # tx, ty가 remain보다 크다면, 이번 프레임은 벽과 부딪힐 일 없음

        ball_start = (ball.x, ball.y)
        ball_end = (ball.x + ball.vx * dt, ball.y + ball.vy * dt)

        agent_start = _lerp(agent_prev, agent_next, elapsed)
        agent_end = _lerp(agent_prev, agent_next, elapsed + dt)


        hit_t = _collision_time(
            agent_start, agent_end, agent_r,
            ball_start, ball_end, ball.r
        )
        if hit_t is not None:
            # ball_end까지 다 이동시키지 않고, 실제로 닿은 지점(hit_t)에서 멈춘다.
            ball.x, ball.y = _lerp(ball_start, ball_end, hit_t)
            # hit_t는 이 sub-step 안에서의 시점이므로, agent 보정용으로는
            # elapsed를 더해 프레임 전체 기준 절대 시점으로 바꿔서 돌려준다.
            return True, elapsed + hit_t * dt

        ball.x, ball.y = ball_end
        elapsed += dt

        if dt == remain: break

        # 벽과 부딪혔다면 반사
        if tx == dt: ball.vx *= -1
        if ty == dt: ball.vy *= -1

    return False, None

# 직선 이동에 대한 collision. 충돌 시 이번 sub-step 안에서 닿은 시점(hit_t, 0~1)을,
# 충돌이 없으면 None을 반환한다.
def _collision_time(agent_prev, agent_next, agent_r, ball_prev, ball_next, ball_r):
    prev_agent_x, prev_agent_y = agent_prev
    prev_ball_x, prev_ball_y = ball_prev
    next_agent_x, next_agent_y = agent_next
    next_ball_x, next_ball_y = ball_next

    px, py = prev_ball_x - prev_agent_x, prev_ball_y - prev_agent_y
    agent_vx, agent_vy = next_agent_x - prev_agent_x, next_agent_y - prev_agent_y
    ball_vx, ball_vy = next_ball_x - prev_ball_x, next_ball_y - prev_ball_y

    R = agent_r + ball_r

    # 상대속도
    vx, vy = ball_vx - agent_vx, ball_vy - agent_vy

    '''
    수식
    previous 상대 위치를 p0, 상대 속도를 v라고 할 때
    0<=t<=1에 대해서, ||p+tv||_2는 agent와 ball의 상대거리.
    ||p+tv||_2^2 = R^2  
    -> <v,v> t^2 + 2 <p,v> t + (<p,p> - R^2) = 0  (t에 대한 2차방정식)
    a = <v,v>, b = 2 <p,v>, c = <p,p> - R^2
    '''

    a = vx**2 + vy**2
    b = 2 * (px * vx + py * vy)
    c = px**2 + py**2 - R**2

    if c <= 0: return 0.0  # frame 시작시 원 내부인가?
    if a == 0: return None  # 상대 위치가 그대로임.


    discriminant = b**2 - 4*a*c

    if discriminant < 0: return None  # 실수 해가 없음 -> 충돌 없음
    else:
        hit_t = (-b - discriminant**(1/2)) / (2*a)  # 근의 공식
        if 0 <= hit_t <= 1: return hit_t  # 이번 프레임 내에서 충돌 있음
        else: return None

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
    def __init__(self, seed=None): 
        self.rng = random.Random(seed)
        self.state = GameState()
        self.reset()
        
    def reset(self, seed=None):
        if seed is not None:
            self.rng.seed(seed)
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
        agent_next = (state.agent.x, state.agent.y)

        for ball in state.balls:
            is_collision, hit_t = _move_ball(ball, agent_prev, agent_next, state.agent.r)
            if is_collision:
                # agent도 공과 같은 충돌 시점에서 멈춘 위치로 되돌림
                state.agent.x, state.agent.y = _lerp(agent_prev, agent_next, hit_t)
                state.alive = False
                
                

        if state.steps % config.BALL_ADD_FREQUENCY == 0:
            _spawn_balls(state.balls, self.rng)
                    
        state.steps += 1

        if state.steps % config.SCORE_INTERVAL == 0:
            state.score += 1

        
        #print(len(state.balls))
