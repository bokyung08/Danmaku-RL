import torch

from env import DanmakuVecEnv

# (k, flip): k번 반시계 회전 후 좌우 반전 여부
D4_TRANSFORMS = (
    (0, False),  # 원본
    (0, True),   # 좌우 반전
    (1, False),  # 반시계 90도
    (1, True),   # 반시계 90도 + 좌우 반전
    (2, False),  # 180도
    (2, True),   # 상하 반전
    (3, False),  # 시계 90도
    (3, True),   # 시계 90도 + 좌우 반전
)

# 공이 항상 좌상단에서 옴
# D4 transform을 할 경우 agent는 공이 네 모서리에서 나온다고 착각할 수 있음
SPAWN_SAFE_TRANSFORMS = (
    (0, False),  # identity
    (3, True),   # (x, y) -> (y, x)
)

TRANSFORM_MODES = {
    "d4": D4_TRANSFORMS,
    "spawn_safe": SPAWN_SAFE_TRANSFORMS,
}

ACTION_TO_VEC = {
    0: ( 0,  0),  # stop
    1: ( 0, -1),  # up
    2: ( 0,  1),  # down
    3: (-1,  0),  # left
    4: ( 1,  0),  # right
    5: (-1, -1),  # up-left
    6: ( 1, -1),  # up-right
    7: (-1,  1),  # down-left
    8: ( 1,  1),  # down-right
}
VEC_TO_ACTION = {v: k for k, v in ACTION_TO_VEC.items()}  # 역 변환


def _rotate_xy(x, y, rotation):
    """(x, y)를 반시계 90도 * rotation 만큼 회전한다.
    """
    for _ in range(rotation % 4):
        x, y = y, -x
    return x, y


def transform_img_obs(obs, rotation, horizontal_flip):
    """DanmakuImgEnv 관측 (..., C, H, W)에 D4 대칭(회전/반전)을 적용한다."""
    obs = torch.rot90(
        obs,
        k=rotation,
        dims=(-2, -1)
    )
    if horizontal_flip:
        obs = torch.flip(obs, dims=(-1,))  # 좌우반전

    return obs


def transform_vec_obs(obs, rotation, horizontal_flip):
    """DanmakuVecEnv 용 (AGENT_FEATURE_NUM + BALL_FEATURE_NUM*N)
    agent: [x,y], ball: [x,y,vx,vy,mask] * N
    모두 [-1, 1]로 정규화 되어있음 mask 유지하고 나머지에 적용
    """
    n_agent_feat = DanmakuVecEnv.AGENT_FEATURE_NUM
    n_ball_feat = DanmakuVecEnv.BALL_FEATURE_NUM
    n_balls = (obs.shape[-1] - n_agent_feat) // n_ball_feat

    agent = obs[..., :n_agent_feat]  # (B, 2)
    balls = obs[..., n_agent_feat:].reshape(*obs.shape[:-1], n_balls, n_ball_feat)  # (B, 40, 5)

    # x, y, v에 대해 회전
    agent_x, agent_y = _rotate_xy(agent[..., 0], agent[..., 1], rotation)
    ball_x, ball_y = _rotate_xy(balls[..., 0], balls[..., 1], rotation)
    ball_vx, ball_vy = _rotate_xy(balls[..., 2], balls[..., 3], rotation)

    # x, y, v에 좌우 반전
    if horizontal_flip:
        agent_x, ball_x, ball_vx = -agent_x, -ball_x, -ball_vx

    agent_out = torch.stack([agent_x, agent_y], dim=-1)  # (B, 2)
    balls_out = torch.stack(
        [ball_x, ball_y, ball_vx, ball_vy, balls[..., 4]], dim=-1
    )  # (B, 40, 5),  (B, 40) 5개를 -1 축에 쌓기 때문

    # obs.shape[:-1] = (B,)  # 앞에 차원이 더 쌓일 상황을 대비. 현재는 obs.shape[0]과 같음
    # (B, 2) + (B, 200) => (B, 202)
    return torch.cat([agent_out, balls_out.reshape(*obs.shape[:-1], -1)], dim=-1)


def transform_action(action, rotation, horizontal_flip):
    dx, dy = ACTION_TO_VEC[action]
    dx, dy = _rotate_xy(dx, dy, rotation)

    # 좌우 반전
    if horizontal_flip:
        dx = -dx

    return VEC_TO_ACTION[(dx, dy)]

def transform_actions(actions, rotation, horizontal_flip):
    lookup = torch.tensor(
        [
            transform_action(action, rotation, horizontal_flip)
            for action in range(9)
        ],
        dtype=actions.dtype,
        device=actions.device,
    )

    return lookup[actions]


def augment_transitions(obs, actions, next_obs, augmentation_mode="d4"):
    # 이미지 관측은 (B, C, H, W) (4차원), 벡터 관측은 (B, feature) (2차원) 이라
    # 차원 수로 구분한다 (agent.py가 모델을 고를 때 쓰는 것과 같은 기준).
    is_image_obs = obs.dim() > 2
    transform = transform_img_obs if is_image_obs else transform_vec_obs
    transforms = TRANSFORM_MODES[augmentation_mode]

    batch_size = obs.shape[0]
    transform_ids = torch.randint(
        0,
        len(transforms),
        (batch_size,),
        device=obs.device,
    )

    obs_variants = []
    next_obs_variants = []
    action_variants = []

    # 미리 만들어두기
    for rotation, horizontal_flip in transforms:
        obs_variants.append(
            transform(obs, rotation, horizontal_flip)
        )
        next_obs_variants.append(
            transform(next_obs, rotation, horizontal_flip)
        )
        action_variants.append(
            transform_actions(actions, rotation, horizontal_flip)
        )

    # (B, 8, ...), (B, 8)
    obs_variants = torch.stack(obs_variants, dim=1)
    next_obs_variants = torch.stack(next_obs_variants, dim=1)
    action_variants = torch.stack(action_variants, dim=1)

    batch_indices = torch.arange(batch_size, device=obs.device)

    return (
        obs_variants[batch_indices, transform_ids],
        action_variants[batch_indices, transform_ids],
        next_obs_variants[batch_indices, transform_ids],
    )
