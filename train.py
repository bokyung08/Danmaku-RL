import argparse
import random
import sys
import time

import numpy as np
import torch

# Windows 콘솔은 cp949라서 인코딩 불가 문자 하나로 학습 전체가 죽을 수 있다.
# 인코딩은 그대로 두고 실패 시 대체 문자로 넘어가게만 한다.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="backslashreplace")

from config import (
    SEED,
    AGENT_TYPE,
    LR,
    N_EPISODES,
    START_EPS,
    EPS_DECAY,
    FINAL_EPS,
    GAMMA,
    HIDDEN_SIZE,
    LEARNING_STARTS,
    TRAIN_FREQUENCY,
    TARGET_NETWORK_FREQUENCY,
    BUFFER_CAPACITY,
    BATCH_SIZE,
    EVAL_EPISODES,
    EVAL_INTERVAL,
    LOG_INTERVAL,
    OUTPUT_ROOT,
)
from env import DanmakuImgEnv
from agent import DQNAgent, DDQNAgent, evaluate
from experiment_results import (
    make_experiment_path,
    save_experiment_results,
    save_checkpoint,
)


# 평가는 매번 같은 시드 집합을 쓴다. 그래야 체크포인트 간 비교가 짝지은 비교가 되어
# 작은 개선도 노이즈와 구분할 수 있다.
EVAL_SEED_BASE = 20_000


def create_environment():
    # DanmakuImgEnv는 gym.Env가 아니므로 gym.make / RecordEpisodeStatistics를 쓰지 않는다.
    # 에피소드 통계는 학습 루프에서 직접 수집한다.
    return DanmakuImgEnv()


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_agent(env, agent_type, device):
    if agent_type not in ("DQN", "DDQN"):
        raise ValueError(f"Invalid Model Name: {agent_type}")

    agent_class = DQNAgent if agent_type == "DQN" else DDQNAgent
    print(f"{agent_type} device: {device}")
    return agent_class(
        env=env,
        learning_rate=LR,
        initial_epsilon=START_EPS,
        epsilon_decay=EPS_DECAY,
        final_epsilon=FINAL_EPS,
        discount_factor=GAMMA,
        hidden_size=HIDDEN_SIZE,
        batch_size=BATCH_SIZE,
        learning_starts=LEARNING_STARTS,
        train_frequency=TRAIN_FREQUENCY,
        target_network_frequency=TARGET_NETWORK_FREQUENCY,
        capacity=BUFFER_CAPACITY,
        device=device,
    )


def train_agent(env, agent_type, seed, experiment_path):
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = create_agent(env, agent_type, device)

    eval_env = create_environment()  # 평가는 학습 env의 상태를 건드리지 않도록 분리
    episode_records = []
    eval_history = []
    best_mean_score = -float("inf")
    total_decision_steps = 0
    start_time = time.time()

    for episode in range(N_EPISODES):
        episode_seed = seed if episode == 0 else None
        state, _ = env.reset(seed=episode_seed)
        done = False

        episode_return = 0.0
        episode_steps = 0
        loss_index_at_start = len(agent.training_error)
        q_index_at_start = len(agent.q_values)

        while not done:
            action = agent.get_action(state)
            action = int(action)
            next_state, reward, terminated, truncated, info = env.step(action)

            agent.rb.add((state, action, reward, terminated, next_state))
            agent.update()

            state = next_state
            done = terminated or truncated
            episode_return += reward
            episode_steps += 1

        agent.decay_epsilon()
        total_decision_steps += episode_steps

        episode_losses = agent.training_error[loss_index_at_start:]
        episode_q_values = agent.q_values[q_index_at_start:]
        episode_records.append({
            "episode": episode,
            "decision_steps": episode_steps,
            "physics_steps": int(info["steps"]),
            "score": int(info["score"]),
            "ep_return": episode_return,
            "epsilon": agent.epsilon,
            "mean_loss": float(np.mean(episode_losses)) if episode_losses else "",
            "mean_q": float(np.mean(episode_q_values)) if episode_q_values else "",
        })

        if (episode + 1) % LOG_INTERVAL == 0:
            recent = episode_records[-LOG_INTERVAL:]
            elapsed = time.time() - start_time
            recent_losses = [row["mean_loss"] for row in recent if row["mean_loss"] != ""]
            recent_q_values = [row["mean_q"] for row in recent if row["mean_q"] != ""]
            print(
                f"ep {episode + 1}/{N_EPISODES} "
                f"steps={total_decision_steps} "
                f"score{LOG_INTERVAL}={np.mean([r['score'] for r in recent]):.2f} "
                f"eps={agent.epsilon:.3f} "
                f"loss={np.mean(recent_losses) if recent_losses else float('nan'):.5f} "
                f"q={np.mean(recent_q_values) if recent_q_values else float('nan'):.3f} "
                f"sps={total_decision_steps / max(elapsed, 1e-9):.0f} "
                f"elapsed={elapsed / 60:.1f}m",
                flush=True,
            )

        if (episode + 1) % EVAL_INTERVAL == 0 or (episode + 1) == N_EPISODES:
            metrics = evaluate(
                agent, eval_env, num_episodes=EVAL_EPISODES, seed=EVAL_SEED_BASE
            )
            eval_history.append({
                "episode": episode + 1,
                "env_steps": total_decision_steps,
                **{key: metrics[key] for key in (
                    "mean_score", "median_score", "std_score", "sem_score",
                    "max_score", "p_ge_120", "p_ge_60", "mean_length", "action_entropy",
                )},
            })
            print(
                f"  [eval] ep={episode + 1} "
                f"mean={metrics['mean_score']:.2f}+-{metrics['sem_score']:.2f} "
                f"median={metrics['median_score']:.1f} max={metrics['max_score']} "
                f"P(>=120)={metrics['p_ge_120']:.2f} "
                f"entropy={metrics['action_entropy']:.2f}/2.20",
                flush=True,
            )
            # 행동 엔트로피가 0에 가까우면 한 행동만 내보내는 정책 붕괴 상태다.
            if metrics["action_entropy"] < 0.3:
                print("  [warn] 행동 엔트로피 < 0.3 nats : 정책 붕괴 의심", flush=True)

            if metrics["mean_score"] > best_mean_score:
                best_mean_score = metrics["mean_score"]
                save_checkpoint(
                    experiment_path / "best.pt",
                    agent,
                    {"episode": episode + 1, "eval": metrics},
                )

    eval_env.close()
    return agent, episode_records, eval_history, time.time() - start_time


def run_experiment(seed, agent_type=AGENT_TYPE, eval_episodes=EVAL_EPISODES):
    experiment_path = make_experiment_path(
        agent_type=agent_type,
        seed=seed,
        output_root=OUTPUT_ROOT,
    )
    print(f"output: {experiment_path}", flush=True)

    env = create_environment()

    # 학습 전에 random 정책 baseline을 같은 시드 집합으로 재둔다.
    random_baseline = evaluate(
        None, env, num_episodes=eval_episodes, seed=EVAL_SEED_BASE
    )
    print(
        f"random baseline: mean={random_baseline['mean_score']:.2f}"
        f"+-{random_baseline['sem_score']:.2f} max={random_baseline['max_score']}",
        flush=True,
    )

    agent, episode_records, eval_history, elapsed = train_agent(
        env, agent_type, seed, experiment_path
    )

    final_eval = evaluate(agent, env, num_episodes=eval_episodes, seed=EVAL_SEED_BASE)

    output_directory = save_experiment_results(
        experiment_path=experiment_path,
        config_values=_get_experiment_config(seed, agent_type, eval_episodes),
        episode_records=episode_records,
        eval_history=eval_history,
        training_errors=agent.training_error,
        random_baseline=random_baseline,
        final_eval=final_eval,
        agent=agent,
        elapsed_seconds=elapsed,
    )
    env.close()

    return output_directory, random_baseline, final_eval


def _get_experiment_config(seed, agent_type, eval_episodes):
    return {
        "SEED": seed,
        "AGENT_TYPE": agent_type,
        "LR": LR,
        "N_EPISODES": N_EPISODES,
        "START_EPS": START_EPS,
        "EPS_DECAY": EPS_DECAY,
        "FINAL_EPS": FINAL_EPS,
        "GAMMA": GAMMA,
        "LEARNING_STARTS": LEARNING_STARTS,
        "TRAIN_FREQUENCY": TRAIN_FREQUENCY,
        "TARGET_NETWORK_FREQUENCY": TARGET_NETWORK_FREQUENCY,
        "HIDDEN_SIZE": HIDDEN_SIZE,
        "BUFFER_CAPACITY": BUFFER_CAPACITY,
        "BATCH_SIZE": BATCH_SIZE,
        "EVAL_EPISODES": eval_episodes,
        "EVAL_INTERVAL": EVAL_INTERVAL,
        "EVAL_SEED_BASE": EVAL_SEED_BASE,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Danmaku DQN/DDQN agent")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--agent-type",
        choices=("DQN", "DDQN"),
        default=AGENT_TYPE,
    )
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--n-episodes", type=int, default=N_EPISODES)
    parser.add_argument("--start-eps", type=float, default=START_EPS)
    parser.add_argument("--eps-decay", type=float, default=None)
    parser.add_argument("--final-eps", type=float, default=FINAL_EPS)
    parser.add_argument("--gamma", type=float, default=GAMMA)

    parser.add_argument("--learning-starts", type=int, default=LEARNING_STARTS)
    parser.add_argument("--train-frequency", type=int, default=TRAIN_FREQUENCY)
    parser.add_argument(
        "--target-network-frequency",
        type=int,
        default=TARGET_NETWORK_FREQUENCY,
    )
    parser.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE)
    parser.add_argument("--buffer-capacity", type=int, default=BUFFER_CAPACITY)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)

    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL)
    parser.add_argument("--log-interval", type=int, default=LOG_INTERVAL)
    parser.add_argument("--output-root", default=OUTPUT_ROOT)

    return parser.parse_args()


def apply_args(args):
    global SEED, AGENT_TYPE
    global LR, N_EPISODES, START_EPS, EPS_DECAY, FINAL_EPS, GAMMA
    global LEARNING_STARTS, TRAIN_FREQUENCY, TARGET_NETWORK_FREQUENCY
    global HIDDEN_SIZE, BUFFER_CAPACITY, BATCH_SIZE
    global EVAL_EPISODES, EVAL_INTERVAL, LOG_INTERVAL, OUTPUT_ROOT

    SEED = args.seed
    AGENT_TYPE = args.agent_type

    LR = args.lr
    N_EPISODES = args.n_episodes
    START_EPS = args.start_eps
    FINAL_EPS = args.final_eps
    GAMMA = args.gamma
    # --n-episodes를 바꾸면 decay도 따라가야 하므로 명시 지정이 없으면 재계산한다.
    # config.py와 동일하게 전체 에피소드의 10% 지점에서 FINAL_EPS에 도달하도록 맞춘다.
    EPS_DECAY = (
        args.eps_decay if args.eps_decay is not None
        else START_EPS / max(N_EPISODES / 10, 1)
    )

    LEARNING_STARTS = args.learning_starts
    TRAIN_FREQUENCY = args.train_frequency
    TARGET_NETWORK_FREQUENCY = args.target_network_frequency
    HIDDEN_SIZE = args.hidden_size
    BUFFER_CAPACITY = args.buffer_capacity
    BATCH_SIZE = args.batch_size

    EVAL_EPISODES = args.eval_episodes
    EVAL_INTERVAL = args.eval_interval
    LOG_INTERVAL = args.log_interval
    OUTPUT_ROOT = args.output_root


def main():
    args = parse_args()
    apply_args(args)

    output_directory, random_baseline, final_eval = run_experiment(
        seed=SEED,
        agent_type=AGENT_TYPE,
        eval_episodes=EVAL_EPISODES,
    )

    print(f"Saved to: {output_directory}")
    print(
        f"Random : mean={random_baseline['mean_score']:.2f} "
        f"median={random_baseline['median_score']:.1f} "
        f"max={random_baseline['max_score']}"
    )
    print(
        f"Learned: mean={final_eval['mean_score']:.2f}"
        f"+-{final_eval['sem_score']:.2f} "
        f"median={final_eval['median_score']:.1f} "
        f"max={final_eval['max_score']} "
        f"P(>=120)={final_eval['p_ge_120']:.2f} "
        f"entropy={final_eval['action_entropy']:.2f}"
    )


if __name__ == "__main__":
    main()
