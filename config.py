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
AGENT_TYPE = 'DQN'  # DQN, or DDQN. 학습이 어느 정도 될 때까지는 DQN을 기본값으로 둔다.
LR = 2.5e-4                # 이미지 CNN 기준.
N_EPISODES = 2000
START_EPS = 1.0         
FINAL_EPS = 0.01   
GAMMA = 0.99        
DATA_AUGMENTATION = False 
AUGMENTATION_MODE = "spawn_safe"  # "spawn_safe" or 8방향
HIDDEN_SIZE = 256
LAYER_NORM = False
LOSS_FN = 'mse'  # 'huber'(=smooth L1) or 'mse'
MAX_GRAD_NORM = 10.0  # gradient clipping 임계값. 0 이하로 두면 clipping을 끈다
DUELING_NET = False  # Dueling DQN 사용 여부. Dueling은 Q(s,a) = V(s) + A(s,a) 로 Q를 분리하여 학습하는 방법

OUTPUT_ROOT = "results"
GIF_SEED = 20_000       # best checkpoint 시각화에 사용할 고정 seed


# Attention (vector obs 전용)
USE_ATTENTION = False
ATTENTION_NUM_HEADS = 4  # HIDDEN_SIZE가 이 값으로 나누어져야 함. 1이면 single-head와 동일
ATTENTION_FUSION = "residual"  # "residual" or historical "concat"
ATTENTION_POSITION_MODE = "relative"  # "relative" or historical "absolute"

# For DQN, DDQN
LEARNING_STARTS = 8000
TRAIN_FREQUENCY = 4  # k step 마다 1 gradient update
TARGET_NETWORK_FREQUENCY = 1000  # TRAIN_FREQUENCY의 배수로 유지하는게 좋음
BUFFER_CAPACITY = 100000  # 메모리 = CAPACITY * 2 * 4 * 84 * 84 바이트 (10000이면 약 564 MB)
BATCH_SIZE = 32
# 평가 / 로깅
EVAL_EPISODES = 20      # 평가 에피소드 수
EVAL_INTERVAL = 100     # 몇 에피소드마다 평가할지
LOG_INTERVAL = 20       # 몇 에피소드마다 진행 상황을 출력할지

# PQN (replay buffer, target network 없이 병렬 환경 + LayerNorm으로 학습을 안정화하는 DQN 변형)
# https://docs.cleanrl.dev/rl-algorithms/pqn/
PQN_NUM_ENVS = 128              # 동시에 굴릴 환경 개수
PQN_NUM_STEPS = 32              # 환경 하나당 한 iteration에 모으는 rollout 길이
PQN_TOTAL_TIMESTEPS = 1_000_000   # 전체 학습 스텝 수 (= NUM_ENVS * NUM_STEPS * iteration 수)
PQN_NUM_MINIBATCHES = 32
PQN_UPDATE_EPOCHS = 2           # 모은 rollout 하나를 몇 번 반복해서 학습할지
PQN_Q_LAMBDA = 0.65
PQN_ANNEAL_LR = False
PQN_EXPLORATION_FRACTION = 0.10  # 전체 스텝 중 몇 %에 걸쳐 epsilon을 줄일지
PQN_EVAL_INTERVAL = 100          # 몇 iteration마다 평가할지
PQN_LOG_INTERVAL = 1            # 몇 iteration마다 진행 상황을 출력할지

# Game config
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
ENV_TYPE = "img"  # "img" or "vec"
N_FRAME_SKIP = 4  # 하나의 Action 반복 횟수
MAX_TIME_STEPS = 10800  # 3 minutes game playing
SEED = None  # None or int, None일 경우 실행할 때마다 다른 실행 결과

# for img env
N_FRAME_STACK = 4 # observation에 쌓을 프레임 수 -> 4장을 하나의 state 로

# HUD / 렌더링
HUD_FONT_SIZE = 28
HUD_MARGIN = (10, 10)

AGENT_COLOR = (80, 200, 255)
BALL_COLOR = (255, 90, 90)
BACKGROUND_COLOR = (15, 15, 25)
HUD_COLOR = (230, 230, 230)
