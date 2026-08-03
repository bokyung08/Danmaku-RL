import math

# 화면
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600
PHYSICS_FPS = 60
RENDER_FPS = 60 
GAME_SPEED = 1.0 
MAX_FRAME_TIME = 0.25 

# 에이전트
AGENT_RADIUS = 10
AGENT_SPEED = 5
# 대각선 이동 시 축별 속도. 직선 이동과 전체 이동 속도(유클리드 거리)를 맞추기 위해
# AGENT_SPEED 를 sqrt(2) 로 나눈다.
AGENT_DIAG_SPEED = AGENT_SPEED / math.sqrt(2)
BALL_RADIUS = 5
MIN_BALL_SPEED = 1
MAX_BALL_SPEED = 15

# 학습
AGENT_TYPE = 'DDQN'  # DQN, or DDQN. 학습이 어느 정도 될 때까지는 DQN을 기본값으로 둔다.
LR = 1e-4                # 이미지 CNN 기준. frozenlake의 1e-3은 10배 큼
N_EPISODES = 100_000
START_EPS = 1.0         # Start with 100% random actions
EPS_DECAY = START_EPS / (N_EPISODES / 10)  # Reduce exploration over time. e.g. in here, eps_decay = 0.0005
FINAL_EPS = 0.01   
GAMMA = 0.99        
DA = False  # data augmentation (flip)  

# For DQN, DDQN
LEARNING_STARTS = 4000
TRAIN_FREQUENCY = 4  # k step 마다 1 gradient update
TARGET_NETWORK_FREQUENCY = 1000  # TRAIN_FREQUENCY의 배수로 유지할 것
HIDDEN_SIZE = 512    # Nature CNN의 FC 크기
BUFFER_CAPACITY = 100000  # 메모리 = CAPACITY * 2 * 4 * 84 * 84 바이트 (10000이면 약 564 MB)
BATCH_SIZE = 32

# 평가 / 로깅
EVAL_EPISODES = 20      # 평가 에피소드 수
EVAL_INTERVAL = 100     # 몇 에피소드마다 평가할지
LOG_INTERVAL = 20       # 몇 에피소드마다 진행 상황을 출력할지
OUTPUT_ROOT = "results"


MAX_BALL_NUM = 40
SCORE_INTERVAL = PHYSICS_FPS
BALL_ADD_FREQUENCY = PHYSICS_FPS * 3 

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



# Env config
N_FRAME_STACK = 4 # observation에 쌓을 프레임 수 -> 4장을 하나의 state 로 
N_FRAME_SKIP = 4 # 하나의 Action 반복 횟수
MAX_TIME_STEPS = 10800  # 3 minutes game playing
SEED = None  # None or int, None일 경우 실행할 때마다 다른 실행 결과

# HUD / 렌더링
HUD_FONT_SIZE = 28
HUD_MARGIN = (10, 10)

AGENT_COLOR = (80, 200, 255)
BALL_COLOR = (255, 90, 90)
BACKGROUND_COLOR = (15, 15, 25)
HUD_COLOR = (230, 230, 230)