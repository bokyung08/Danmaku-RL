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
    FINAL_EPS,
    GAMMA,
    HIDDEN_SIZE,
    LAYER_NORM,
    LEARNING_STARTS,
    TRAIN_FREQUENCY,
    TARGET_NETWORK_FREQUENCY,
    BUFFER_CAPACITY,
    BATCH_SIZE,
    LOSS_FN,
    MAX_GRAD_NORM,
    DUELING_NET,
    USE_ATTENTION,
    ATTENTION_NUM_HEADS,
    ATTENTION_FUSION,
    ATTENTION_POSITION_MODE,
    EVAL_EPISODES,
    EVAL_INTERVAL,
    LOG_INTERVAL,
    OUTPUT_ROOT,
    PHYSICS_FPS,
    N_FRAME_STACK,
    N_FRAME_SKIP,
    MAX_TIME_STEPS,
    GIF_SEED,
    DATA_AUGMENTATION,
    AUGMENTATION_MODE,
    ENV_TYPE,
)
from env import DanmakuImgEnv, DanmakuVecEnv
from agent import DQNAgent, DDQNAgent, evaluate
from experiment_results import (
    make_experiment_path,
    save_experiment_results,
    save_checkpoint,
    save_intermediate_results,
)
from metric import srank, weight_norm
from record_agent_gif import record


# 평가는 매번 같은 시드 집합을 쓴다. 그래야 체크포인트 간 비교가 짝지은 비교가 되어
# 작은 개선도 노이즈와 구분할 수 있다.
EVAL_SEED_BASE = 20_000


def create_environment(env_type="img"):
    if env_type == "img":
        return DanmakuImgEnv()
    elif env_type == "vec":
        return DanmakuVecEnv(normalize="none" if USE_ATTENTION else "near")
    else:
        raise ValueError(f"Invalid environment type: {env_type}")

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
    return agent_class(
        env=env,
        learning_rate=LR,
        initial_epsilon=START_EPS,
        epsilon_decay=EPS_DECAY,
        final_epsilon=FINAL_EPS,
        discount_factor=GAMMA,
        hidden_size=HIDDEN_SIZE,
        layer_norm=LAYER_NORM,
        batch_size=BATCH_SIZE,
        learning_starts=LEARNING_STARTS,
        train_frequency=TRAIN_FREQUENCY,
        target_network_frequency=TARGET_NETWORK_FREQUENCY,
        capacity=BUFFER_CAPACITY,
        device=device,
        loss_fn=LOSS_FN,
        max_grad_norm=MAX_GRAD_NORM,
        dueling_net=DUELING_NET,
        use_attention=USE_ATTENTION,
        num_heads=ATTENTION_NUM_HEADS,
        attention_fusion=ATTENTION_FUSION,
        attention_position_mode=ATTENTION_POSITION_MODE,
    )


def print_training_setup(agent, env, agent_type, seed, device):
    trainable_parameters = sum(
        parameter.numel()
        for parameter in agent.model.parameters()
        if parameter.requires_grad
    )
    print(
        "\n[training setup]\n"
        f"model   : {agent_type} / {agent.model.__class__.__name__} "
        f"({trainable_parameters:,} trainable parameters)\n"
        f"arch    : \n            {agent.model.describe()}\n"
        f"run     : device={device}, seed={seed}, episodes={N_EPISODES}\n"
        f"optim   : lr={LR:g}, gamma={GAMMA:g}, batch={BATCH_SIZE}, "
        f"buffer={BUFFER_CAPACITY}, hidden={HIDDEN_SIZE}, layer_norm={LAYER_NORM}, "
        f"dueling_net={DUELING_NET}, use_attention={USE_ATTENTION}"
        f"({ATTENTION_NUM_HEADS} heads, fusion={ATTENTION_FUSION}, "
        f"position={ATTENTION_POSITION_MODE}), "
        f"augmentation={DATA_AUGMENTATION}({AUGMENTATION_MODE}), "
        f"loss_fn={LOSS_FN}, max_grad_norm={MAX_GRAD_NORM:g}\n"
        f"update  : learning_starts={LEARNING_STARTS}, train_every={TRAIN_FREQUENCY}, "
        f"target_every={TARGET_NETWORK_FREQUENCY}\n"
        f"explore : epsilon={START_EPS:g}->{FINAL_EPS:g}, decay={EPS_DECAY:g}\n"
        f"env     : type={ENV_TYPE}, obs={env.observation_shape}, actions={env.n_actions}, "
        f"frame_stack={N_FRAME_STACK}, frame_skip={N_FRAME_SKIP}, "
        f"max_steps={MAX_TIME_STEPS}\n"
        f"logging : log_every={LOG_INTERVAL}, eval_every={EVAL_INTERVAL}, "
        f"eval_episodes={EVAL_EPISODES}\n"
        f"artifact: best_agent.gif + best_agent_observation.gif, seed={GIF_SEED}, "
        f"model_fps={PHYSICS_FPS / N_FRAME_SKIP:g}, decision_frames=True\n",
        flush=True,
    )


def train_agent(env, agent_type, seed, experiment_path):
    set_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = create_agent(env, agent_type, device)
    print_training_setup(agent, env, agent_type, seed, device)

    eval_env = create_environment(env_type=ENV_TYPE)  # 평가는 학습 env의 상태를 건드리지 않도록 분리
    episode_records = []
    eval_history = []
    metric_history = []
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

            replay_next_state = (
                info.get("final_observation", next_state)
                if truncated
                else next_state
            )
            agent.rb.add((state, action, reward, terminated, replay_next_state))
            agent.update(
                data_augmentation=DATA_AUGMENTATION,
                augmentation_mode=AUGMENTATION_MODE,
            )

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

        # eval 혹은 실험이 다 끝났을 때
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

            current_weight_norm = weight_norm(agent.model)
            current_srank = float("nan")
            srank_max = 0
            if len(agent.rb) >= HIDDEN_SIZE:
                metric_batch_size = HIDDEN_SIZE
                # Metric evaluation must not alter future replay sampling.
                cpu_rng_state = torch.random.get_rng_state()
                try:
                    metric_batch = agent.rb.sample(metric_batch_size)
                finally:
                    torch.random.set_rng_state(cpu_rng_state)

                with torch.no_grad():
                    metric_features = agent.model.forward_features(
                        metric_batch.observations
                    )
                current_srank = srank(
                    metric_features,
                    tau=0.01,
                )
                srank_max = min(metric_features.shape)

            metric_history.append({
                "episode": episode + 1,
                "env_steps": total_decision_steps,
                "weight_norm": current_weight_norm,
                "srank": current_srank,
                "srank_max": srank_max,
            })

            print(
                f"  [eval] ep={episode + 1} "
                f"mean={metrics['mean_score']:.2f}+-{metrics['sem_score']:.2f} "
                f"median={metrics['median_score']:.1f} max={metrics['max_score']} "
                f"P(>=120)={metrics['p_ge_120']:.2f} "
                f"entropy={metrics['action_entropy']:.2f}/2.20 "
                f"wnorm={current_weight_norm:.3f} "
                f"srank={current_srank}/{srank_max}",
                flush=True,  # 버퍼에 저장하지 않고 바로 내보냄
            )
            # 한 행동만 하는지 체크
            if metrics["action_entropy"] < 0.3:
                print("  [warn] 행동 엔트로피 < 0.3", flush=True)

            if metrics["mean_score"] > best_mean_score:
                best_mean_score = metrics["mean_score"]
                save_checkpoint(
                    experiment_path / "best.pt",
                    agent,
                    {
                        "episode": episode + 1,
                        "eval": metrics,
                        "config": _get_experiment_config(
                            seed, agent_type, EVAL_EPISODES
                        ),
                    },
                )

            save_intermediate_results(
                experiment_path=experiment_path,
                config_values=_get_experiment_config(
                    seed, agent_type, EVAL_EPISODES
                ),
                episode_records=episode_records,
                eval_history=eval_history,
                metric_history=metric_history,
                agent=agent,
                elapsed_seconds=time.time() - start_time,
            )

    eval_env.close()
    return agent, episode_records, eval_history, metric_history, time.time() - start_time


def run_experiment(seed, agent_type=AGENT_TYPE, eval_episodes=EVAL_EPISODES):
    experiment_path = make_experiment_path(
        agent_type=agent_type,
        seed=seed,
        output_root=OUTPUT_ROOT,
    )
    print(f"output: {experiment_path}", flush=True)

    env = create_environment(env_type=ENV_TYPE)

    # 학습 전에 random 정책 baseline을 같은 시드 집합으로 재둔다.
    random_baseline = evaluate(
        None, env, num_episodes=eval_episodes, seed=EVAL_SEED_BASE
    )
    print(
        f"random baseline: mean={random_baseline['mean_score']:.2f}"
        f"+-{random_baseline['sem_score']:.2f} max={random_baseline['max_score']}",
        flush=True,
    )

    agent, episode_records, eval_history, metric_history, elapsed = train_agent(
        env, agent_type, seed, experiment_path
    )

    final_eval = evaluate(agent, env, num_episodes=eval_episodes, seed=EVAL_SEED_BASE)

    output_directory = save_experiment_results(
        experiment_path=experiment_path,
        config_values=_get_experiment_config(seed, agent_type, eval_episodes),
        episode_records=episode_records,
        eval_history=eval_history,
        metric_history=metric_history,
        training_errors=agent.training_error,
        random_baseline=random_baseline,
        final_eval=final_eval,
        agent=agent,
        elapsed_seconds=elapsed,
    )
    env.close()
    del agent

    checkpoint_path = output_directory / "best.pt"
    gif_path = output_directory / "best_agent.gif"
    try:
        print(f"creating GIF from: {checkpoint_path}", flush=True)
        # make gif from the best model
        record(
            checkpoint=checkpoint_path,
            output=gif_path,
            seed=GIF_SEED,
        )
    except Exception as error:
        # Checkpoints and metrics are already safely stored at this point.
        print(f"[warn] GIF creation failed: {error}", flush=True)

    return output_directory, random_baseline, final_eval


def _get_experiment_config(seed, agent_type, eval_episodes):
    return {
        "SEED": seed,
        "ENV_TYPE": ENV_TYPE,
        "VEC_OBS_NORMALIZE": "none" if USE_ATTENTION else "near",
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
        "LAYER_NORM": LAYER_NORM,
        "DATA_AUGMENTATION": DATA_AUGMENTATION,
        "AUGMENTATION_MODE": AUGMENTATION_MODE,
        "LOSS_FN": LOSS_FN,
        "MAX_GRAD_NORM": MAX_GRAD_NORM,
        "DUELING_NET": DUELING_NET,
        "USE_ATTENTION": USE_ATTENTION,
        "ATTENTION_NUM_HEADS": ATTENTION_NUM_HEADS,
        "ATTENTION_FUSION": ATTENTION_FUSION,
        "ATTENTION_POSITION_MODE": ATTENTION_POSITION_MODE,
        "BUFFER_CAPACITY": BUFFER_CAPACITY,
        "BATCH_SIZE": BATCH_SIZE,
        "EVAL_EPISODES": eval_episodes,
        "EVAL_INTERVAL": EVAL_INTERVAL,
        "EVAL_SEED_BASE": EVAL_SEED_BASE,
        "GIF_SEED": GIF_SEED,
        "GIF_FPS": PHYSICS_FPS / N_FRAME_SKIP,
        "GIF_MATCH_MODEL_FRAMES": True,
        "GIF_OBSERVATION_VIEW": "observation[-1], 84x84 grayscale",
        "N_FRAME_SKIP": N_FRAME_SKIP,
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
    parser.add_argument(
        "--layer-norm",
        action=argparse.BooleanOptionalAction,
        default=LAYER_NORM,
        help="NatureCNN convolution/feature layers에 LayerNorm 적용",
    )
    parser.add_argument(
        "--dueling-net",
        action=argparse.BooleanOptionalAction,
        default=DUELING_NET,
        help="Dueling network의 value/advantage head 사용",
    )
    parser.add_argument(
        "--use-attention",
        action=argparse.BooleanOptionalAction,
        default=USE_ATTENTION,
        help="vector obs에서 MLP 대신 agent-query/ball-key,value cross-attention 사용",
    )
    parser.add_argument("--attention-num-heads", type=int, default=ATTENTION_NUM_HEADS)
    parser.add_argument(
        "--attention-fusion",
        choices=("residual", "concat"),
        default=ATTENTION_FUSION,
    )
    parser.add_argument(
        "--attention-position-mode",
        choices=("relative", "absolute"),
        default=ATTENTION_POSITION_MODE,
    )
    parser.add_argument(
        "--data-augmentation",
        action=argparse.BooleanOptionalAction,
        default=DATA_AUGMENTATION,
        help="transition마다 augmentation 적용",
    )
    parser.add_argument(
        "--augmentation-mode",
        choices=("spawn_safe", "d4"),
        default=AUGMENTATION_MODE,
        help="observation augmentation symmetry group",
    )
    parser.add_argument("--buffer-capacity", type=int, default=BUFFER_CAPACITY)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--loss-fn", choices=("huber", "mse"), default=LOSS_FN)
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=MAX_GRAD_NORM,
        help="gradient clipping 임계값. 0이면 clipping 없음",
    )

    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--eval-interval", type=int, default=EVAL_INTERVAL)
    parser.add_argument("--log-interval", type=int, default=LOG_INTERVAL)
    parser.add_argument("--output-root", default=OUTPUT_ROOT)

    parser.add_argument("--env-type", choices=("img", "vec"), default=ENV_TYPE)

    return parser.parse_args()


def apply_args(args):
    global SEED, AGENT_TYPE
    global LR, N_EPISODES, START_EPS, EPS_DECAY, FINAL_EPS, GAMMA
    global LEARNING_STARTS, TRAIN_FREQUENCY, TARGET_NETWORK_FREQUENCY
    global HIDDEN_SIZE, LAYER_NORM, DATA_AUGMENTATION, AUGMENTATION_MODE
    global BUFFER_CAPACITY, BATCH_SIZE, LOSS_FN, MAX_GRAD_NORM, DUELING_NET
    global USE_ATTENTION, ATTENTION_NUM_HEADS, ATTENTION_FUSION
    global ATTENTION_POSITION_MODE
    global EVAL_EPISODES, EVAL_INTERVAL, LOG_INTERVAL, OUTPUT_ROOT
    global ENV_TYPE

    SEED = args.seed
    AGENT_TYPE = args.agent_type
    ENV_TYPE = args.env_type

    LR = args.lr
    N_EPISODES = args.n_episodes
    START_EPS = args.start_eps
    FINAL_EPS = args.final_eps
    GAMMA = args.gamma
    # 10%의 episode 동안 선형 감소 일단 default로 유지.
    EPS_DECAY = (
        args.eps_decay
        if args.eps_decay is not None
        else (START_EPS - FINAL_EPS) / max(N_EPISODES * 0.1, 1)
    )

    LEARNING_STARTS = args.learning_starts
    TRAIN_FREQUENCY = args.train_frequency
    TARGET_NETWORK_FREQUENCY = args.target_network_frequency
    HIDDEN_SIZE = args.hidden_size
    LAYER_NORM = args.layer_norm
    DATA_AUGMENTATION = args.data_augmentation
    AUGMENTATION_MODE = args.augmentation_mode
    BUFFER_CAPACITY = args.buffer_capacity
    BATCH_SIZE = args.batch_size
    LOSS_FN = args.loss_fn
    MAX_GRAD_NORM = args.max_grad_norm
    DUELING_NET = args.dueling_net
    USE_ATTENTION = args.use_attention
    ATTENTION_NUM_HEADS = args.attention_num_heads
    ATTENTION_FUSION = args.attention_fusion
    ATTENTION_POSITION_MODE = args.attention_position_mode

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
