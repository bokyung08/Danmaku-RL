# 탄막 회피 시뮬레이터 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** CLAUDE.md 명세대로 결정적(deterministic)이고 렌더링과 분리된 탄막 회피 게임 시뮬레이터를 구현한다.

**Architecture:** `game.py` 는 pygame 을 전혀 import 하지 않는 순수 로직(Agent/Ball/GameState + reset/step)만 담당한다. `level.py` 는 레벨별 공 스폰 테이블 조회와 레벨업 판정을 캡슐화한다. `render.py` 는 GameState 를 읽기 전용으로 그리기만 하며 상태를 변경하지 않는다. `main.py` 는 pygame 이벤트 루프에서 키 입력을 action(0~4)으로 변환해 `Game.step()` 을 호출한다.

**Tech Stack:** Python 3.11, pygame(렌더링/입력), pytest(테스트), 표준 venv.

## Global Constraints

- `game.py` 에는 `pygame` 을 절대 import 하지 않는다. (CLAUDE.md §0)
- 화면 그리기는 `render.py`, 키 입력은 `main.py` 가 전담한다. (CLAUDE.md §0)
- 모든 상수는 `config.py` 한 곳에 모은다. 다른 파일에 매직 넘버를 두지 않는다. (CLAUDE.md §0)
- 난이도(공 개수/속도/크기)는 랜덤이 아니라 `config.py` 의 `LEVEL_SPAWNS` 리스트로 미리 정의한다. (CLAUDE.md §0, §9)
- 코드 내 `print` 메시지는 한국어로 작성한다. (CLAUDE.md §0)
- 같은 행동 시퀀스를 넣으면 항상 같은 결과가 나와야 한다(결정성). (CLAUDE.md §1)
- 렌더링 없이 헤드리스로도 `step` 을 반복 호출해 동작해야 한다. (CLAUDE.md §1)

---

## Task 0: 프로젝트 환경 설정

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`

**Interfaces:**
- Produces: 이후 모든 태스크가 사용할 venv(`.venv`)와 `pygame`, `pytest` 설치.

- [ ] **Step 1: git 저장소 초기화**

```bash
cd /Users/mac/Desktop/dodge
git init
```

- [ ] **Step 2: venv 생성 및 활성화**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

- [ ] **Step 3: requirements.txt 작성**

```
pygame==2.6.1
pytest==8.3.3
```

- [ ] **Step 4: 의존성 설치**

```bash
pip install -r requirements.txt
```

Expected: `pygame` 과 `pytest` 가 오류 없이 설치됨.

- [ ] **Step 5: .gitignore 작성**

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
```

- [ ] **Step 6: 커밋**

```bash
git add requirements.txt .gitignore
git commit -m "chore: 프로젝트 환경 설정 (venv, pygame, pytest)"
```

---

## Task 1: config.py — 상수 및 레벨 테이블

**Files:**
- Create: `config.py`

**Interfaces:**
- Produces:
  - `SCREEN_WIDTH`, `SCREEN_HEIGHT`, `FPS`
  - `AGENT_RADIUS`, `AGENT_SPEED`
  - `LEVEL_SPAWNS: dict[int, list[tuple[int,int,int,int,int]]]`
  - `LEVEL_UP_STEPS: dict[int, int]`
  - `MAX_LEVEL: int`
  - `ACTION_STOP=0, ACTION_UP=1, ACTION_DOWN=2, ACTION_LEFT=3, ACTION_RIGHT=4`
  - `PHASE_READY="READY", PHASE_PLAYING="PLAYING", PHASE_GAMEOVER="GAMEOVER"`
- 이후 모든 파일이 이 상수들을 import 해서 사용한다. 이 파일 외에는 매직 넘버를 두지 않는다.

- [ ] **Step 1: config.py 작성**

```python
"""전역 상수 및 레벨 테이블. 이 파일 외에는 매직 넘버를 두지 않는다."""

# 화면
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
FPS = 60

# 에이전트
AGENT_RADIUS = 10
AGENT_SPEED = 5

# 행동(action) 정의
ACTION_STOP = 0
ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4

# 게임 진행 단계
PHASE_READY = "READY"
PHASE_PLAYING = "PLAYING"
PHASE_GAMEOVER = "GAMEOVER"

# 레벨별 공 스폰 정의
# 각 공: (초기 x, 초기 y, vx, vy, 반지름)
LEVEL_SPAWNS = {
    1: [(20, 20, 4, 3, 8)],
    2: [(20, 20, 5, 3, 8),
        (20, 20, 3, 5, 8)],
    3: [(20, 20, 5, 4, 10),
        (20, 20, 4, 5, 10),
        (20, 20, 6, 2, 10)],
    4: [(20, 20, 6, 4, 10),
        (20, 20, 4, 6, 10),
        (20, 20, 6, 3, 10),
        (20, 20, 3, 6, 10)],
    5: [(20, 20, 7, 5, 12),
        (20, 20, 5, 7, 12),
        (20, 20, 7, 4, 12),
        (20, 20, 4, 7, 12),
        (20, 20, 6, 6, 12)],
}

# 레벨 전환 기준 (해당 프레임 생존 시 다음 레벨)
LEVEL_UP_STEPS = {1: 500, 2: 800, 3: 1200, 4: 1600}

MAX_LEVEL = max(LEVEL_SPAWNS.keys())
```

- [ ] **Step 2: import 확인**

```bash
python3 -c "import config; print(config.MAX_LEVEL, len(config.LEVEL_SPAWNS))"
```

Expected: `5 5` 출력, 예외 없음.

- [ ] **Step 3: 커밋**

```bash
git add config.py
git commit -m "feat: config.py 상수 및 레벨 테이블 정의"
```

---

## Task 2: level.py — 스폰 조회 및 레벨 전환 판정

**Files:**
- Create: `level.py`
- Test: `tests/test_level.py`

**Interfaces:**
- Consumes: `config.LEVEL_SPAWNS`, `config.LEVEL_UP_STEPS`, `config.MAX_LEVEL`
- Produces:
  - `get_spawns(level: int) -> list[tuple[int,int,int,int,int]]` — 정의된 최고 레벨을 넘으면 마지막 레벨 테이블을 반환.
  - `next_level(level: int, steps: int) -> int` — 기준 스텝 이상 생존 시 `level + 1`, 아니면 `level` 유지. 최고 레벨에서는 항상 `level` 유지.
  - `game.py` 의 `_spawn_balls`, `Game.step` 이 이 두 함수를 사용한다.

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tests/test_level.py
import config
import level


def test_get_spawns_returns_level_table():
    assert level.get_spawns(1) == config.LEVEL_SPAWNS[1]
    assert level.get_spawns(3) == config.LEVEL_SPAWNS[3]


def test_get_spawns_clamps_to_max_level():
    assert level.get_spawns(99) == config.LEVEL_SPAWNS[config.MAX_LEVEL]


def test_next_level_stays_below_threshold():
    assert level.next_level(1, config.LEVEL_UP_STEPS[1] - 1) == 1


def test_next_level_advances_at_exact_threshold():
    assert level.next_level(1, config.LEVEL_UP_STEPS[1]) == 2


def test_next_level_stays_at_max_level():
    assert level.next_level(config.MAX_LEVEL, 999999) == config.MAX_LEVEL
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
pytest tests/test_level.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'level'`

- [ ] **Step 3: level.py 구현**

```python
"""레벨별 공 스폰 테이블 조회 및 레벨 전환 판정."""
import config


def get_spawns(level):
    """레벨에 해당하는 공 스폰 리스트를 반환한다.

    정의된 최고 레벨을 넘으면 마지막 레벨 상태를 유지한다.
    """
    clamped = min(level, config.MAX_LEVEL)
    return list(config.LEVEL_SPAWNS[clamped])


def next_level(level, steps):
    """생존 스텝 수가 기준 이상이면 다음 레벨 번호를 반환한다."""
    threshold = config.LEVEL_UP_STEPS.get(level)
    if threshold is not None and steps >= threshold:
        return level + 1
    return level
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
pytest tests/test_level.py -v
```

Expected: PASS (5 passed)

- [ ] **Step 5: 커밋**

```bash
git add level.py tests/test_level.py
git commit -m "feat: level.py 스폰 조회 및 레벨 전환 판정 구현"
```

---

## Task 3: game.py — 순수 게임 로직 (Agent/Ball/GameState/reset/step)

**Files:**
- Create: `game.py`
- Test: `tests/test_game.py`

**Interfaces:**
- Consumes: `config.*`, `level.get_spawns`, `level.next_level`
- Produces:
  - `Agent(x, y, radius=config.AGENT_RADIUS, speed=config.AGENT_SPEED)` dataclass
  - `Ball(x, y, vx, vy, radius)` dataclass
  - `GameState` — 속성: `agent`, `balls`, `steps`, `level`, `alive`, `phase`
  - `Game` — `Game()` 생성 시 자동으로 `reset()` 호출. `game.state`(`GameState`), `game.reset()`, `game.step(action)` 제공.
  - `render.py` 의 `Renderer.draw(game)` 은 `game.state` 만 읽는다 (Task 4에서 사용).
  - `main.py` 는 `Game()`, `game.state.phase`, `game.step(action)`, `game.reset()` 을 사용한다 (Task 5에서 사용).

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tests/test_game.py
import config
from game import Game


def test_reset_sets_initial_state():
    game = Game()
    state = game.state
    assert state.agent.x == config.SCREEN_WIDTH / 2
    assert state.agent.y == config.SCREEN_HEIGHT / 2
    assert state.level == 1
    assert state.steps == 0
    assert state.alive is True
    assert state.phase == config.PHASE_PLAYING
    assert len(state.balls) == len(config.LEVEL_SPAWNS[1])


def test_determinism_same_actions_same_result():
    actions = [config.ACTION_RIGHT, config.ACTION_UP, config.ACTION_STOP, config.ACTION_DOWN] * 50

    game_a = Game()
    for action in actions:
        game_a.step(action)

    game_b = Game()
    for action in actions:
        game_b.step(action)

    assert game_a.state.agent.x == game_b.state.agent.x
    assert game_a.state.agent.y == game_b.state.agent.y
    assert game_a.state.steps == game_b.state.steps
    assert [(b.x, b.y, b.vx, b.vy) for b in game_a.state.balls] == \
        [(b.x, b.y, b.vx, b.vy) for b in game_b.state.balls]


def test_agent_clipped_to_screen_bounds():
    game = Game()
    for _ in range(200):
        game.step(config.ACTION_LEFT)
    assert game.state.agent.x == game.state.agent.radius

    game.reset()
    for _ in range(200):
        game.step(config.ACTION_UP)
    assert game.state.agent.y == game.state.agent.radius


def test_wall_reflection_keeps_ball_inside_bounds():
    game = Game()
    for _ in range(300):
        game.step(config.ACTION_STOP)
        for ball in game.state.balls:
            assert ball.radius <= ball.x <= config.SCREEN_WIDTH - ball.radius
            assert ball.radius <= ball.y <= config.SCREEN_HEIGHT - ball.radius


def test_collision_ends_game_immediately():
    game = Game()
    ball = game.state.balls[0]
    ball.vx = 0
    ball.vy = 0
    game.state.agent.x = ball.x
    game.state.agent.y = ball.y

    game.step(config.ACTION_STOP)

    assert game.state.alive is False
    assert game.state.phase == config.PHASE_GAMEOVER


def test_step_ignored_after_gameover():
    game = Game()
    game.state.phase = config.PHASE_GAMEOVER
    steps_before = game.state.steps

    game.step(config.ACTION_RIGHT)

    assert game.state.steps == steps_before


def test_level_transitions_at_exact_step_threshold():
    game = Game()
    for _ in range(config.LEVEL_UP_STEPS[1]):
        game.step(config.ACTION_STOP)

    assert game.state.level == 2
    assert len(game.state.balls) == len(config.LEVEL_SPAWNS[2])


def test_headless_runs_many_steps_without_render():
    game = Game()
    for _ in range(2000):
        if game.state.phase != config.PHASE_PLAYING:
            break
        game.step(config.ACTION_STOP)
    assert game.state.steps >= 0
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
pytest tests/test_game.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'game'`

- [ ] **Step 3: game.py 구현**

```python
"""순수 게임 로직. pygame 을 절대 import 하지 않는다."""
from dataclasses import dataclass, field

import config
import level


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


def _spawn_balls(lvl):
    return [Ball(x, y, vx, vy, r) for (x, y, vx, vy, r) in level.get_spawns(lvl)]


def _move_agent(agent, action):
    if action == config.ACTION_UP:
        agent.y -= agent.speed
    elif action == config.ACTION_DOWN:
        agent.y += agent.speed
    elif action == config.ACTION_LEFT:
        agent.x -= agent.speed
    elif action == config.ACTION_RIGHT:
        agent.x += agent.speed


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

        new_level = level.next_level(state.level, state.steps)
        if new_level != state.level:
            state.level = new_level
            state.balls = _spawn_balls(state.level)
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
pytest tests/test_game.py -v
```

Expected: PASS (8 passed)

- [ ] **Step 5: 헤드리스 물리 검증 로그 (CLAUDE.md §14 3번 — 렌더링 전 물리 검증)**

```bash
python3 -c "
import config
from game import Game

game = Game()
for i in range(10):
    game.step(config.ACTION_STOP)
    b = game.state.balls[0]
    print(f'step={game.state.steps} ball=({b.x:.1f},{b.y:.1f}) agent=({game.state.agent.x:.1f},{game.state.agent.y:.1f})')
"
```

Expected: 10줄의 좌표 로그가 예외 없이 출력됨.

- [ ] **Step 6: 커밋**

```bash
git add game.py tests/test_game.py
git commit -m "feat: game.py 순수 게임 로직 구현 (reset/step, 벽반사, 충돌, 레벨전환)"
```

---

## Task 4: render.py — pygame 렌더링

**Files:**
- Create: `render.py`
- Test: `tests/test_render.py`

**Interfaces:**
- Consumes: `config.*`, `game.Game` (읽기 전용으로 `game.state` 만 사용)
- Produces: `Renderer()` — `Renderer.draw(game)` 메서드. 상태를 변경하지 않는다.
- `main.py` 는 `Renderer()` 를 생성하고 매 프레임 `renderer.draw(game)` 을 호출한다 (Task 5에서 사용).

- [ ] **Step 1: 실패하는 테스트 작성**

```python
# tests/test_render.py
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import config
from game import Game
from render import Renderer


def test_draw_does_not_mutate_state():
    game = Game()
    game.step(config.ACTION_RIGHT)
    renderer = Renderer()

    def snapshot():
        s = game.state
        return (s.agent.x, s.agent.y, s.steps, s.level, s.phase,
                [(b.x, b.y, b.vx, b.vy) for b in s.balls])

    before = snapshot()
    renderer.draw(game)
    renderer.draw(game)
    after = snapshot()

    assert before == after


def test_draw_gameover_screen_does_not_raise():
    game = Game()
    game.state.phase = config.PHASE_GAMEOVER
    renderer = Renderer()

    renderer.draw(game)
```

- [ ] **Step 2: 테스트 실패 확인**

```bash
SDL_VIDEODRIVER=dummy pytest tests/test_render.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'render'`

- [ ] **Step 3: render.py 구현**

```python
"""pygame 렌더링 전담. 상태를 그리기만 하고 변경하지 않는다."""
import pygame

import config

AGENT_COLOR = (80, 200, 255)
BALL_COLOR = (255, 90, 90)
BACKGROUND_COLOR = (15, 15, 25)
HUD_COLOR = (230, 230, 230)
GAMEOVER_COLOR = (255, 255, 255)


class Renderer:
    def __init__(self):
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
        pygame.display.set_caption("탄막 회피 시뮬레이터")
        self.font = pygame.font.SysFont(None, 28)

    def draw(self, game):
        state = game.state
        self.screen.fill(BACKGROUND_COLOR)

        pygame.draw.circle(
            self.screen, AGENT_COLOR,
            (int(state.agent.x), int(state.agent.y)), state.agent.radius,
        )
        for ball in state.balls:
            pygame.draw.circle(
                self.screen, BALL_COLOR,
                (int(ball.x), int(ball.y)), ball.radius,
            )

        hud = self.font.render(
            f"레벨: {state.level}  생존 스텝: {state.steps}  공 개수: {len(state.balls)}",
            True, HUD_COLOR,
        )
        self.screen.blit(hud, (10, 10))

        if state.phase == config.PHASE_GAMEOVER:
            self._draw_gameover(state)

        pygame.display.flip()

    def _draw_gameover(self, state):
        over = self.font.render("게임 오버", True, GAMEOVER_COLOR)
        score = self.font.render(f"최종 점수: {state.steps}", True, GAMEOVER_COLOR)
        self.screen.blit(
            over,
            (config.SCREEN_WIDTH / 2 - over.get_width() / 2,
             config.SCREEN_HEIGHT / 2 - 20),
        )
        self.screen.blit(
            score,
            (config.SCREEN_WIDTH / 2 - score.get_width() / 2,
             config.SCREEN_HEIGHT / 2 + 10),
        )
```

- [ ] **Step 4: 테스트 통과 확인**

```bash
SDL_VIDEODRIVER=dummy pytest tests/test_render.py -v
```

Expected: PASS (2 passed)

- [ ] **Step 5: 커밋**

```bash
git add render.py tests/test_render.py
git commit -m "feat: render.py pygame 렌더링 구현 (Agent/Ball/HUD/GAMEOVER)"
```

---

## Task 5: main.py — 사람 플레이 루프

**Files:**
- Create: `main.py`

**Interfaces:**
- Consumes: `config.*`, `game.Game`, `render.Renderer`
- Produces: `main()` 진입점. `python3 main.py` 로 실행 가능.

- [ ] **Step 1: main.py 구현**

```python
"""사람 플레이 루프 진입점. 키 입력 처리를 전담한다."""
import pygame

import config
from game import Game
from render import Renderer

KEY_ACTION_MAP = {
    pygame.K_UP: config.ACTION_UP,
    pygame.K_DOWN: config.ACTION_DOWN,
    pygame.K_LEFT: config.ACTION_LEFT,
    pygame.K_RIGHT: config.ACTION_RIGHT,
}


def read_input():
    keys = pygame.key.get_pressed()
    for key, action in KEY_ACTION_MAP.items():
        if keys[key]:
            return action
    return config.ACTION_STOP


def is_restart_key(event):
    return event.type == pygame.KEYDOWN and event.key == pygame.K_r


def main():
    game = Game()
    renderer = Renderer()
    clock = pygame.time.Clock()

    running = True
    while running:
        action = read_input()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if game.state.phase == config.PHASE_GAMEOVER and is_restart_key(event):
                game.reset()

        if game.state.phase == config.PHASE_PLAYING:
            game.step(action)

        renderer.draw(game)
        clock.tick(config.FPS)

    pygame.quit()
    print("게임을 종료합니다.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 헤드리스 import 스모크 테스트 (pygame 초기화 없이 문법/의존성만 확인)**

```bash
SDL_VIDEODRIVER=dummy python3 -c "import main; print('main.py import 성공')"
```

Expected: `main.py import 성공` 출력, 예외 없음.

- [ ] **Step 3: 실제 창을 띄워 수동 플레이 검증 (사람이 직접 확인)**

```bash
python3 main.py
```

수동 확인 항목 (CLAUDE.md §13 검증 체크리스트):
- 방향키로 Agent 가 이동하고 화면 밖으로 나가지 않는다.
- 공이 벽에서 튕기며 파고들거나 떨지 않는다.
- 공에 닿는 순간 정확히 GAMEOVER 화면과 최종 점수가 표시된다.
- `R` 키로 GAMEOVER 상태에서 재시작된다.
- 레벨업 시점(500/800/1200/1600 스텝)에 공 개수가 늘어난다.
- 창을 닫으면 "게임을 종료합니다." 가 한국어로 출력된다.

- [ ] **Step 4: 커밋**

```bash
git add main.py
git commit -m "feat: main.py 사람 플레이 루프 구현"
```

---

## Task 6: 전체 검증 체크리스트 재확인

**Files:**
- 없음 (기존 파일 전체 재검증)

- [ ] **Step 1: 전체 테스트 스위트 실행**

```bash
SDL_VIDEODRIVER=dummy pytest -v
```

Expected: 모든 테스트 PASS.

- [ ] **Step 2: CLAUDE.md §13 검증 체크리스트와 대조**

| 항목 | 검증 방법 | 결과 |
|------|-----------|------|
| 결정성 | `tests/test_game.py::test_determinism_same_actions_same_result` | |
| 벽 반사 | `tests/test_game.py::test_wall_reflection_keeps_ball_inside_bounds` | |
| 충돌 | `tests/test_game.py::test_collision_ends_game_immediately` | |
| 경계 | `tests/test_game.py::test_agent_clipped_to_screen_bounds` | |
| 레벨 전환 | `tests/test_game.py::test_level_transitions_at_exact_step_threshold` | |
| 헤드리스 | `tests/test_game.py::test_headless_runs_many_steps_without_render` | |

- [ ] **Step 3: 최종 커밋 (필요 시)**

```bash
git status
```

Expected: 미커밋 변경사항 없음.
