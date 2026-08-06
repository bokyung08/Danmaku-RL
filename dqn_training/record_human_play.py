import argparse
import sys
from pathlib import Path

import numpy as np
import pygame

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import config
from env import DanmakuVecEnv
from render import Renderer
from human import get_key, move

DEMO_PATH = THIS_DIR / "human_demos.npz"

# train_dqn.py의 reward shaping과 반드시 동일하게 유지 (train_dqn.py의 5. 파라미터 설정 참고)
gamma = 0.999
survival_reward = 0.01
collision_penalty = -1.0
completion_reward = 1.0
distance_pixels = 100.0
distance_ratio = 0.1
stall_penalty = 0.002


def distance_reward(env):
    state = env.game.state
    if not state.balls:
        return 0.0
    agent = state.agent
    distance_min = min(
        ((ball.x - agent.x) ** 2 + (ball.y - agent.y) ** 2) ** 0.5 - ball.r - agent.r for ball in state.balls
    )
    return float(np.clip(distance_min / distance_pixels, 0.0, 1.0))


def play_and_record(env, renderer, clock, min_survival_seconds):
    fixed_decision_dt = config.N_FRAME_SKIP / config.PHYSICS_FPS
    accumulator = 0.0
    state, _ = env.reset(seed=config.SEED)

    # 저장이 확정된(임계값을 넘긴) 에피소드만 모으는 리스트
    kept_observations, kept_actions, kept_rewards = [], [], []
    kept_next_observations, kept_terminated, kept_truncated = [], [], []

    # 현재 진행 중인 한 판의 임시 버퍼 (판이 끝나야 저장 여부가 결정됨)
    episode_observations, episode_actions, episode_rewards = [], [], []
    episode_next_observations, episode_terminated, episode_truncated = [], [], []

    episode_count = 0
    kept_count = 0

    while True:
        frame_dt = clock.tick(config.RENDER_FPS) / 1000.0
        frame_dt = min(frame_dt, config.MAX_FRAME_TIME)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return kept_observations, kept_actions, kept_rewards, kept_next_observations, kept_terminated, kept_truncated
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return kept_observations, kept_actions, kept_rewards, kept_next_observations, kept_terminated, kept_truncated

        if env.game.state.alive:
            accumulator += frame_dt * config.GAME_SPEED
            while accumulator >= fixed_decision_dt and env.game.state.alive:
                action = move(*get_key())

                previous_agent_x = env.game.state.agent.x
                previous_agent_y = env.game.state.agent.y
                previous_dist = distance_reward(env)

                next_state, _reward, terminated, truncated, info = env.step(action)
                finished = terminated or truncated

                next_dist = 0.0 if finished else distance_reward(env)
                agent = env.game.state.agent
                moved = (agent.x != previous_agent_x) or (agent.y != previous_agent_y)

                if terminated:
                    base_reward = collision_penalty
                elif truncated:
                    base_reward = completion_reward
                else:
                    base_reward = survival_reward
                dist_reward = distance_ratio * (gamma * next_dist - previous_dist)  # potential based reward shaping
                stall = bool(env.game.state.balls) and not moved  # 공이 있는데 움직이지 않음
                learning_reward = base_reward + dist_reward - (stall_penalty if stall else 0.0)

                episode_observations.append(state)
                episode_actions.append(action)
                episode_rewards.append(learning_reward)
                episode_next_observations.append(next_state)
                episode_terminated.append(terminated)
                episode_truncated.append(truncated)

                state = next_state
                accumulator -= fixed_decision_dt

                if finished:
                    episode_count += 1
                    survival_seconds = info["steps"] / config.PHYSICS_FPS
                    if survival_seconds >= min_survival_seconds:
                        kept_count += 1
                        kept_observations.extend(episode_observations)
                        kept_actions.extend(episode_actions)
                        kept_rewards.extend(episode_rewards)
                        kept_next_observations.extend(episode_next_observations)
                        kept_terminated.extend(episode_terminated)
                        kept_truncated.extend(episode_truncated)
                        verdict = "saved"
                    else:
                        verdict = f"discarded (< {min_survival_seconds:.1f}s)"
                    print(f"episode {episode_count}: survived {survival_seconds:.1f}s, score {info['score']} -> {verdict}")
                    episode_observations, episode_actions, episode_rewards = [], [], []
                    episode_next_observations, episode_terminated, episode_truncated = [], [], []
        else:
            accumulator = 0.0
            state, _ = env.reset(seed=config.SEED)  # 죽으면 자동으로 다음 판 시작

        renderer.draw(env.game)


def save_demo(observations, actions, rewards, next_observations, terminated_flags, truncated_flags):
    new_observations = np.asarray(observations, dtype=np.float32)
    new_next_observations = np.asarray(next_observations, dtype=np.float32)
    new_actions = np.asarray(actions, dtype=np.int64)
    new_rewards = np.asarray(rewards, dtype=np.float32)
    new_terminated = np.asarray(terminated_flags, dtype=np.bool_)
    new_truncated = np.asarray(truncated_flags, dtype=np.bool_)

    if DEMO_PATH.exists():
        existing = np.load(DEMO_PATH)
        new_observations = np.concatenate([existing["observations"], new_observations])
        new_next_observations = np.concatenate([existing["next_observations"], new_next_observations])
        new_actions = np.concatenate([existing["actions"], new_actions])
        new_rewards = np.concatenate([existing["rewards"], new_rewards])
        new_terminated = np.concatenate([existing["terminated"], new_terminated])
        new_truncated = np.concatenate([existing["truncated"], new_truncated])

    np.savez(
        DEMO_PATH,
        observations=new_observations,
        next_observations=new_next_observations,
        actions=new_actions,
        rewards=new_rewards,
        terminated=new_terminated,
        truncated=new_truncated,
    )
    print(f"saved {len(new_actions)} transitions total -> {DEMO_PATH}")


def parse_args():
    parser = argparse.ArgumentParser(description="Record human play as DQN demo transitions.")
    parser.add_argument("--min-survival-seconds", type=float, default=5.0)
    return parser.parse_args()


def main():
    args = parse_args()

    env = DanmakuVecEnv()
    renderer = Renderer(render_mode="human")
    clock = pygame.time.Clock()

    print("화살표 키로 조작하세요. 죽으면 자동으로 다음 판이 시작됩니다. ESC 또는 창 닫기로 종료 및 저장.")
    print(f"{args.min_survival_seconds:.1f}초 미만 생존한 판은 저장하지 않습니다 (--min-survival-seconds로 조절 가능).")
    observations, actions, rewards, next_observations, terminated_flags, truncated_flags = play_and_record(
        env, renderer, clock, args.min_survival_seconds
    )

    pygame.quit()

    if actions:
        save_demo(observations, actions, rewards, next_observations, terminated_flags, truncated_flags)
    else:
        print("기록된 경험이 없습니다.")


if __name__ == "__main__":
    main()
