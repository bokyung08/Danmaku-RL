"""순수 게임 로직. pygame 을 절대 import 하지 않는다."""
from dataclasses import dataclass, field

import config


@dataclass
class Agent:
    x: float
    y: float
    radius: int = config.AGENT_RADIUS
    speed: int = config.AGENT_SPEED


@dataclass
class Ball:
    x: float
    y: float
    vx: float
    vy: float
    radius: int


@dataclass
class GameState:
    agent: Agent = None
    balls: list = field(default_factory=list)
    steps: int = 0
    level: int = 1
    alive: bool = True
    phase: str = config.PHASE_READY


def _get_spawns(lvl):
    """레벨에 해당하는 공 스폰 리스트를 반환한다.

    정의된 최고 레벨을 넘으면 마지막 레벨 상태를 유지한다.
    """
    clamped = min(lvl, config.MAX_LEVEL)
    return list(config.LEVEL_SPAWNS[clamped])


def _next_level(lvl, steps):
    """생존 스텝 수가 기준 이상이면 다음 레벨 번호를 반환한다."""
    threshold = config.LEVEL_UP_STEPS.get(lvl)
    if threshold is not None and steps >= threshold:
        return lvl + 1
    return lvl


def _spawn_balls(lvl):
    return [Ball(x, y, vx, vy, r) for (x, y, vx, vy, r) in _get_spawns(lvl)]


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


def _clip_agent(agent):
    agent.x = max(agent.radius, min(config.SCREEN_WIDTH - agent.radius, agent.x))
    agent.y = max(agent.radius, min(config.SCREEN_HEIGHT - agent.radius, agent.y))


def _reflect(ball):
    if ball.x - ball.radius < 0:
        ball.x = ball.radius
        ball.vx = -ball.vx
    elif ball.x + ball.radius > config.SCREEN_WIDTH:
        ball.x = config.SCREEN_WIDTH - ball.radius
        ball.vx = -ball.vx

    if ball.y - ball.radius < 0:
        ball.y = ball.radius
        ball.vy = -ball.vy
    elif ball.y + ball.radius > config.SCREEN_HEIGHT:
        ball.y = config.SCREEN_HEIGHT - ball.radius
        ball.vy = -ball.vy


def _is_collision(agent, ball):
    dx = agent.x - ball.x
    dy = agent.y - ball.y
    r = agent.radius + ball.radius
    return dx * dx + dy * dy <= r * r


class Game:
    def __init__(self):
        self.state = GameState()
        self.reset()

    def reset(self):
        state = self.state
        state.agent = Agent(x=config.SCREEN_WIDTH / 2, y=config.SCREEN_HEIGHT / 2)
        state.level = 1
        state.balls = _spawn_balls(state.level)
        state.steps = 0
        state.alive = True
        state.phase = config.PHASE_PLAYING

    def step(self, action):
        state = self.state
        if state.phase != config.PHASE_PLAYING:
            return

        _move_agent(state.agent, action)
        _clip_agent(state.agent)

        for ball in state.balls:
            ball.x += ball.vx
            ball.y += ball.vy
            _reflect(ball)

        if any(_is_collision(state.agent, ball) for ball in state.balls):
            state.alive = False
            state.phase = config.PHASE_GAMEOVER
            return

        state.steps += 1

        new_level = _next_level(state.level, state.steps)
        if new_level != state.level:
            state.level = new_level
            state.balls += _spawn_balls(state.level)
