import math

# 화면
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
FPS = 60

# 에이전트
AGENT_RADIUS = 10
AGENT_SPEED = 5
# 대각선 이동 시 축별 속도. 직선 이동과 전체 이동 속도(유클리드 거리)를 맞추기 위해
# AGENT_SPEED 를 sqrt(2) 로 나눈다.
AGENT_DIAG_SPEED = AGENT_SPEED / math.sqrt(2)

# 행동(action) 정의
ACTION_STOP = 0
ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_UP_LEFT = 5
ACTION_UP_RIGHT = 6
ACTION_DOWN_LEFT = 7
ACTION_DOWN_RIGHT = 8

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

# HUD / 렌더링
HUD_FONT_NAME = "applesdgothicneo,applegothic,nanumgothic,malgungothic,arial"
HUD_FONT_SIZE = 28
HUD_MARGIN = (10, 10)
GAMEOVER_LINE_OFFSET = 20
