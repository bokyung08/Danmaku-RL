# 탄막 회피 시뮬레이터 개발 명세서

> 강화학습(RL)은 이 단계 범위에서 제외한다. 사람이 직접 플레이하거나 눈/로그로
> 검증할 수 있는 순수 게임 시뮬레이터만 구현한다. 이 시뮬레이터는 이후 RL 환경으로
> 감쌀 수 있도록 결정적(deterministic)이고 렌더링과 분리되어 있어야 한다.

## 0. 구현 규칙 (Claude Code 필독)

- `game.py` 에는 `pygame` 을 절대 import 하지 않는다. 게임 로직은 숫자만 다룬다.
- 화면 그리기는 `render.py`, 키 입력은 `main.py` 가 전담한다.
- 모든 상수는 `config.py` 한 곳에 모은다. 다른 파일에 매직 넘버를 두지 않는다.
- 난이도(공 개수/속도/크기)는 랜덤이 아니라 구체적인 리스트로 미리 정의한다.
- 코드 내 출력(print) 메시지는 한국어로 작성한다.

## 1. 이 단계의 완료 기준

- 키보드(방향키)로 직접 플레이가 가능하다.
- 렌더링 없이 헤드리스로도 `step` 을 반복 호출해 동작한다.
- 같은 행동 시퀀스를 넣으면 항상 같은 결과가 나온다(결정성).

## 2. 파일 구조

```
project/
├── config.py    # 상수, 레벨 테이블, 공 스폰 테이블
├── game.py      # 순수 게임 로직 (렌더링·입력 없음)
├── level.py     # 레벨 파라미터 및 전환 판정
├── render.py    # pygame 렌더링 전담
└── main.py      # 사람 플레이 루프 진입점
```

## 3. 게임 규칙 요약

- 게임 시작 시 Agent 는 화면 중앙에서 시작한다.
- 공은 화면 좌측 상단에서 시작하며 우하향으로 이동한다.
- 공은 벽면에 닿으면 튕긴다(해당 축 속도 반전).
- Agent 가 공에 닿으면 게임이 즉시 종료된다.
- 생존한 프레임 수(steps)가 곧 점수다.

## 4. 상태 정의 (`game.py`)

| 객체 | 속성 | 설명 |
|------|------|------|
| Agent | x, y | 중심 좌표 |
| | radius | 반지름(충돌용) |
| | speed | 프레임당 이동 픽셀 |
| Ball | x, y | 중심 좌표 |
| | vx, vy | 축별 속도 |
| | radius | 반지름 |
| GameState | balls | 공 객체 리스트 |
| | steps | 경과 프레임 수(= 점수) |
| | level | 현재 레벨 |
| | alive | 생존 여부 |
| | phase | READY / PLAYING / GAMEOVER |

## 5. 초기화 로직 (`reset`)

| 대상 | 초기값 |
|------|--------|
| Agent 위치 | 화면 중앙 (W/2, H/2) |
| 공 위치 | 좌측 상단 (반지름만큼 안쪽) |
| 공 속도 | 우하향 (+vx, +vy), 레벨 테이블 값 사용 |
| steps | 0 |
| level | 1 |
| alive | True |
| phase | PLAYING |

## 6. 한 스텝(프레임) 업데이트 순서

`Game.step(action)` 하나가 아래 순서 전체를 캡슐화한다. 순서를 반드시 지킨다.

| 순서 | 처리 |
|------|------|
| 1 | 입력(action)에 따라 Agent 이동 |
| 2 | Agent 경계 클리핑(화면 밖 금지) |
| 3 | 각 공의 위치 갱신 (x += vx, y += vy) |
| 4 | 벽 반사 처리 |
| 5 | 충돌 판정 |
| 6 | 충돌 시 alive=False, phase=GAMEOVER |
| 7 | 생존 시 steps += 1 |
| 8 | 레벨 전환 조건 확인 |

### 행동(action) 정의

| 값 | 행동 |
|----|------|
| 0 | 정지 |
| 1 | 상 |
| 2 | 하 |
| 3 | 좌 |
| 4 | 우 |

## 7. 벽 반사 로직

속도만 반전시키면 공이 벽에 걸쳐 부호가 계속 뒤집히는 떨림이 생긴다. 반드시 위치도
벽 안쪽으로 보정한다.

```python
def reflect(ball, W, H):
    if ball.x - ball.radius < 0:
        ball.x = ball.radius          # 위치 보정
        ball.vx = -ball.vx
    elif ball.x + ball.radius > W:
        ball.x = W - ball.radius
        ball.vx = -ball.vx

    if ball.y - ball.radius < 0:
        ball.y = ball.radius
        ball.vy = -ball.vy
    elif ball.y + ball.radius > H:
        ball.y = H - ball.radius
        ball.vy = -ball.vy
```

## 8. 충돌 판정

원-원 충돌은 중심 거리 제곱과 반지름 합의 제곱을 비교한다(제곱근 회피).

```python
def is_collision(agent, ball):
    dx = agent.x - ball.x
    dy = agent.y - ball.y
    r = agent.radius + ball.radius
    return dx * dx + dy * dy <= r * r
```

## 9. 상수 및 레벨 테이블 (`config.py`)

난이도는 랜덤이 아니라 레벨별 구체 리스트로 정의한다. 매 실행마다 동일한 궤적이
재현되어 검증과 밸런스 조정이 쉽다.

```python
# 화면
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
FPS = 60

# 에이전트
AGENT_RADIUS = 10
AGENT_SPEED = 5

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
```

## 10. 레벨 전환 방식 (`level.py`)

레벨업 시 새 레벨의 공 스폰 리스트를 기존 공 리스트에 추가하는 누적 방식을 사용한다.
레벨이 오를수록 화면에 공이 계속 쌓여 난이도가 누진적으로 상승한다.

- 현재 `steps` 가 `LEVEL_UP_STEPS[level]` 이상이면 `level += 1`.
- 새 레벨의 `LEVEL_SPAWNS[level]` 을 기존 공 리스트에 추가한다(교체가 아니라 누적).
- 정의된 최고 레벨을 넘으면 마지막 레벨 상태를 유지한다(더 이상 공이 추가되지 않는다).

## 11. 렌더링 (`render.py`)

렌더링은 상태를 그리기만 하고 상태를 바꾸지 않는다.

| 요소 | 표시 |
|------|------|
| Agent | 채워진 원(구분되는 색) |
| 공 | 채워진 원 |
| HUD | 현재 레벨, 생존 스텝, 공 개수 |
| GAMEOVER | 화면 중앙에 종료 문구와 최종 점수 |

## 12. 사람 플레이 루프 (`main.py`)

```python
def main():
    game = Game(config)
    renderer = Renderer(config)
    clock = pygame.time.Clock()

    running = True
    while running:
        action = read_input()  # 방향키 -> 0~4

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if game.phase == "GAMEOVER" and is_restart_key(event):
                game.reset()

        if game.phase == "PLAYING":
            game.step(action)

        renderer.draw(game)
        clock.tick(config.FPS)

    print("게임을 종료합니다.")
```

## 13. 검증 체크리스트

| 항목 | 확인 내용 |
|------|-----------|
| 결정성 | 같은 행동 시퀀스 -> 항상 같은 결과 |
| 벽 반사 | 공이 벽에 파고들거나 떨지 않음 |
| 충돌 | 스치는 순간 정확히 종료 |
| 경계 | Agent 가 화면 밖으로 못 나감 |
| 레벨 전환 | 기준 스텝에서 정확히 다음 레벨로 전환 |
| 헤드리스 | 렌더링 없이 step 반복 호출만으로 동작 |

## 14. 권장 구현 순서

1. `config.py` 에 상수와 레벨 테이블 정의
2. `game.py` 에 상태 · `reset` · `step` 구현 (렌더링 없이)
3. 헤드리스로 `step` 을 수백 번 돌려 물리 검증 (print 로 좌표 확인)
4. `render.py` 로 시각화 추가
5. `main.py` 에 사람 플레이 루프 연결
6. `level.py` 분리 및 레벨 전환 확인

> 3번에서 렌더링 전에 물리부터 로그로 검증할 것. 문제 발생 시 물리/렌더링 원인을
> 빠르게 분리할 수 있다.