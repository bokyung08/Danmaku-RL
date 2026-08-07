"""PQN (Parallelised Q-Network) 학습 스크립트.

train.py의 DQN/DDQN과 달리 replay buffer와 target network를 쓰지 않는다. 대신
여러 환경을 동시에(lockstep으로) 진행시켜 rollout을 모으고, 그 rollout으로
Q(lambda) return을 계산한 뒤 여러 epoch에 걸쳐 학습한다. LayerNorm이 target
network를 대신해 학습을 안정화하는 역할을 한다.
참고: https://docs.cleanrl.dev/rl-algorithms/pqn/ (cleanrl의 pqn_atari_envpool.py)

실제 PQN 알고리즘(모델, rollout buffer, Q(lambda) 계산, minibatch 학습)은
agent.py의 PQNAgent에 있다. 이 파일은 병렬 env를 굴리며 PQNAgent를 호출하는
학습 루프, 로깅, 평가, 체크포인트 저장만 담당한다.

env/model/저장 포맷은 train.py와 그대로 호환되도록 맞췄다 (record_agent_gif.py,
experiment_results.py를 수정 없이 그대로 쓴다).
"""

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
    LR,
    GAMMA,
    START_EPS,
    FINAL_EPS,
    HIDDEN_SIZE,
    LAYER_NORM,
    DUELING_NET,
    USE_ATTENTION,
    ATTENTION_NUM_HEADS,
    ATTENTION_FUSION,
    ATTENTION_POSITION_MODE,
    LOSS_FN,
    MAX_GRAD_NORM,
    DATA_AUGMENTATION,
    AUGMENTATION_MODE,
    PQN_NUM_ENVS,
    PQN_NUM_STEPS,
    PQN_TOTAL_TIMESTEPS,
    PQN_NUM_MINIBATCHES,
    PQN_UPDATE_EPOCHS,
    PQN_Q_LAMBDA,
    PQN_ANNEAL_LR,
    PQN_EXPLORATION_FRACTION,
    PQN_EVAL_INTERVAL,
    PQN_LOG_INTERVAL,
    EVAL_EPISODES,
    OUTPUT_ROOT,
    PHYSICS_FPS,
    N_FRAME_STACK,
    N_FRAME_SKIP,
    MAX_TIME_STEPS,
    GIF_SEED,
    ENV_TYPE,
)
from env import DanmakuImgEnv, DanmakuVecEnv
from agent import PQNAgent, evaluate
from experiment_results import (
    make_experiment_path,
    save_experiment_results,
    save_checkpoint,
    save_intermediate_results,
)
from metric import DEFAULT_SRANK_TAU, srank, weight_norm
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


def linear_schedule(start, end, duration, t):
    slope = (end - start) / duration
    return max(slope * t + start, end)


def print_training_setup(agent, envs, num_iterations):
    trainable_parameters = sum(
        parameter.numel() for parameter in agent.model.parameters() if parameter.requires_grad
    )
    batch_size = PQN_NUM_ENVS * PQN_NUM_STEPS
    exploration_steps = int(PQN_TOTAL_TIMESTEPS * PQN_EXPLORATION_FRACTION)
    print(
        "\n[training setup]\n"
        f"model   : PQN / {agent.model.__class__.__name__} ({trainable_parameters:,} trainable parameters)\n"
        f"arch    : \n            {agent.model.describe()}\n"
        f"run     : device={agent.device}, seed={SEED}, total_timesteps={PQN_TOTAL_TIMESTEPS:,}, "
        f"iterations={num_iterations:,}\n"
        f"optim   : optimizer=adam, lr={LR:g}, anneal_lr={PQN_ANNEAL_LR}, "
        f"gamma={GAMMA:g}, hidden={HIDDEN_SIZE}, "
        f"layer_norm={LAYER_NORM}, dueling_net={DUELING_NET}, use_attention={USE_ATTENTION}"
        f"({ATTENTION_NUM_HEADS} heads, fusion={ATTENTION_FUSION}, "
        f"position={ATTENTION_POSITION_MODE}), "
        f"augmentation={DATA_AUGMENTATION}({AUGMENTATION_MODE}), "
        f"loss_fn={LOSS_FN}, max_grad_norm={MAX_GRAD_NORM:g}\n"
        f"rollout : num_envs={PQN_NUM_ENVS}, num_steps={PQN_NUM_STEPS}, batch={batch_size}, "
        f"minibatches={PQN_NUM_MINIBATCHES}, update_epochs={PQN_UPDATE_EPOCHS}, "
        f"q_lambda={PQN_Q_LAMBDA:g}\n"
        f"explore : epsilon={START_EPS:g}->{FINAL_EPS:g} over {exploration_steps:,} steps\n"
        f"env     : type={ENV_TYPE}, obs={envs[0].observation_shape}, actions={envs[0].n_actions}, "
        f"frame_stack={N_FRAME_STACK}, frame_skip={N_FRAME_SKIP}, max_steps={MAX_TIME_STEPS}\n"
        f"logging : log_every={PQN_LOG_INTERVAL} iter, eval_every={PQN_EVAL_INTERVAL} iter, "
        f"eval_episodes={EVAL_EPISODES}\n"
        f"artifact: best_agent.gif, seed={GIF_SEED}\n",
        flush=True,
    )


def train_agent(envs, eval_env, agent, seed, experiment_path):
    set_seed(seed)
    # total_timesteps = 1000만인 경우
    num_envs = len(envs)  # 128
    batch_size = num_envs * PQN_NUM_STEPS  # 4096
    num_iterations = PQN_TOTAL_TIMESTEPS // batch_size  # 1000만 // 4096 = 2441
    exploration_steps = PQN_EXPLORATION_FRACTION * PQN_TOTAL_TIMESTEPS  # 100만

    print_training_setup(agent, envs, num_iterations)

    first_obs = [
        env.reset(seed=seed + i if seed is not None else None)[0]
        for i, env in enumerate(envs)
    ]
    next_obs = torch.as_tensor(np.stack(first_obs), dtype=torch.float32, device=agent.device)
    running_return = np.zeros(num_envs)
    running_length = np.zeros(num_envs)

    episode_records = []
    eval_history = []
    metric_history = []
    best_mean_score = -float("inf")
    episode_count = 0
    global_step = 0
    epsilon = START_EPS
    # 이전 iteration의 학습 결과를 episode_records에 참고용으로 남긴다 (아래 설명 참고).
    latest_loss = ""
    latest_q = ""
    start_time = time.time()

    for iteration in range(num_iterations):
        if PQN_ANNEAL_LR:
            # learning rate: LR -> 0
            agent.set_learning_rate(LR * (1.0 - iteration / max(num_iterations, 1)))

        # --- rollout 수집: num_envs개 환경을 lockstep으로 num_steps만큼 진행 ---
        for step in range(PQN_NUM_STEPS):
            global_step += num_envs
            epsilon = linear_schedule(START_EPS, FINAL_EPS, exploration_steps, global_step)
            actions = agent.act(step, next_obs, epsilon)  # (num_envs, ) 개의 action을 동시에 뽑기 

            next_obs_list = []
            truncated_indices = []
            truncated_final_obs = []
            for i, env in enumerate(envs):  # env를 돌면서 for문으로 step
                obs_i, reward, terminated, truncated, info = env.step(int(actions[i].item()))
                terminated = bool(terminated)
                truncated = bool(truncated)
                done = terminated or truncated

                agent.record_outcome(step, i, reward, terminated, truncated)  # r, term, trun 기록
                running_return[i] += reward
                running_length[i] += 1

                # reset하기 전의 obs_i가 truncation 직전의 실제 마지막 observation
                if truncated:
                    truncated_indices.append(i)
                    truncated_final_obs.append(obs_i)

                if done:
                    episode_count += 1
                    episode_records.append({
                        "episode": episode_count,
                        "decision_steps": int(running_length[i]),  # env.step() 호출 횟수
                        "physics_steps": int(info["steps"]),  # game frame
                        "score": int(info["score"]),
                        "ep_return": float(running_return[i]),
                        "epsilon": epsilon,
                        # loss/Q가 rollout 단위로만 계산되어 episode와 1:1로 안 묶이므로,
                        # 가장 최근 iteration에서 나온 값을 참고용으로 붙임
                        "mean_loss": latest_loss,
                        "mean_q": latest_q,
                    })
                    running_return[i] = 0.0
                    running_length[i] = 0.0
                    obs_i, _ = env.reset()

                next_obs_list.append(obs_i)

            # truncation 직전(reset 전) observation에서 부트스트랩 값을 미리 계산해둔다.
            if truncated_indices:
                # num_envs 개의 원소를 가진 list -> (num of truncated, obs_shape)
                truncated_obs_tensor = torch.as_tensor(
                    np.stack(truncated_final_obs), dtype=torch.float32, device=agent.device,
                )
                agent.bootstrap_truncated(step, truncated_indices, truncated_obs_tensor)

            next_obs = torch.as_tensor(np.stack(next_obs_list), dtype=torch.float32, device=agent.device)

        returns = agent.compute_returns(next_obs)
        latest_loss, latest_q = agent.learn(
            returns, PQN_NUM_MINIBATCHES, PQN_UPDATE_EPOCHS, DATA_AUGMENTATION, AUGMENTATION_MODE
        )

        if (iteration + 1) % PQN_LOG_INTERVAL == 0:
            elapsed = time.time() - start_time
            recent_scores = [row["score"] for row in episode_records[-50:]]
            print(
                f"iter {iteration + 1}/{num_iterations} "
                f"steps={global_step} episodes={episode_count} "
                f"score50={np.mean(recent_scores) if recent_scores else float('nan'):.2f} "
                f"eps={epsilon:.3f} loss={agent.training_error[-1]:.5f} q={agent.q_values[-1]:.3f} "
                f"trunc={int(agent.truncated_buf.sum().item())} "
                f"sps={global_step / max(elapsed, 1e-9):.0f} elapsed={elapsed / 60:.1f}m",
                flush=True,
            )

        if (iteration + 1) % PQN_EVAL_INTERVAL == 0 or (iteration + 1) == num_iterations:
            metrics = evaluate(agent, eval_env, num_episodes=EVAL_EPISODES, seed=EVAL_SEED_BASE)
            eval_history.append({
                "episode": episode_count,
                "env_steps": global_step,
                **{key: metrics[key] for key in (
                    "mean_score", "median_score", "std_score", "sem_score",
                    "max_score", "p_ge_120", "p_ge_60", "mean_length", "action_entropy",
                )},
            })

            current_weight_norm = weight_norm(agent.model)
            metric_features = agent.sample_features(HIDDEN_SIZE)
            current_srank = srank(metric_features, tau=DEFAULT_SRANK_TAU)
            srank_max = min(metric_features.shape)

            metric_history.append({
                "episode": episode_count,
                "env_steps": global_step,
                "weight_norm": current_weight_norm,
                "srank": current_srank,
                "srank_max": srank_max,
            })

            print(
                f"  [eval] iter={iteration + 1} "
                f"mean={metrics['mean_score']:.2f}+-{metrics['sem_score']:.2f} "
                f"median={metrics['median_score']:.1f} max={metrics['max_score']} "
                f"P(>=120)={metrics['p_ge_120']:.2f} "
                f"entropy={metrics['action_entropy']:.2f}/2.20 "
                f"wnorm={current_weight_norm:.3f} "
                f"srank={current_srank}/{srank_max}",
                flush=True,
            )
            if metrics["action_entropy"] < 0.3:
                print("  [warn] 행동 엔트로피 < 0.3 nats : 정책 붕괴 의심", flush=True)

            # mean score 기준으로 best.pt 저장
            if metrics["mean_score"] > best_mean_score:
                best_mean_score = metrics["mean_score"]
                save_checkpoint(
                    experiment_path / "best.pt",
                    agent,
                    {
                        "episode": episode_count,
                        "eval": metrics,
                        "config": _get_experiment_config(seed),
                    },
                )

            save_intermediate_results(
                experiment_path=experiment_path,
                config_values=_get_experiment_config(seed),
                episode_records=episode_records,
                eval_history=eval_history,
                metric_history=metric_history,
                agent=agent,
                elapsed_seconds=time.time() - start_time,
            )

    return episode_records, eval_history, metric_history, time.time() - start_time


def run_experiment(seed):
    experiment_path = make_experiment_path(agent_type="PQN", seed=seed, output_root=OUTPUT_ROOT)
    print(f"output: {experiment_path}", flush=True)

    envs = [create_environment(env_type=ENV_TYPE) for _ in range(PQN_NUM_ENVS)]
    eval_env = create_environment(env_type=ENV_TYPE)

    # 학습 전에 random 정책 baseline을 같은 시드 집합으로 재둔다.
    random_baseline = evaluate(None, eval_env, num_episodes=EVAL_EPISODES, seed=EVAL_SEED_BASE)
    print(
        f"random baseline: mean={random_baseline['mean_score']:.2f}"
        f"+-{random_baseline['sem_score']:.2f} max={random_baseline['max_score']}",
        flush=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = PQNAgent(
        env=envs[0],
        num_envs=PQN_NUM_ENVS,
        num_steps=PQN_NUM_STEPS,
        learning_rate=LR,
        gamma=GAMMA,
        q_lambda=PQN_Q_LAMBDA,
        hidden_size=HIDDEN_SIZE,
        layer_norm=LAYER_NORM,
        dueling_net=DUELING_NET,
        device=device,
        loss_fn=LOSS_FN,
        max_grad_norm=MAX_GRAD_NORM,
        use_attention=USE_ATTENTION,
        num_heads=ATTENTION_NUM_HEADS,
        attention_fusion=ATTENTION_FUSION,
        attention_position_mode=ATTENTION_POSITION_MODE,
    )

    episode_records, eval_history, metric_history, elapsed = train_agent(
        envs, eval_env, agent, seed, experiment_path
    )

    final_eval = evaluate(agent, eval_env, num_episodes=EVAL_EPISODES, seed=EVAL_SEED_BASE)

    output_directory = save_experiment_results(
        experiment_path=experiment_path,
        config_values=_get_experiment_config(seed),
        episode_records=episode_records,
        eval_history=eval_history,
        metric_history=metric_history,
        training_errors=agent.training_error,
        random_baseline=random_baseline,
        final_eval=final_eval,
        agent=agent,
        elapsed_seconds=elapsed,
    )
    for env in envs:
        env.close()
    eval_env.close()
    del agent

    checkpoint_path = output_directory / "best.pt"
    gif_path = output_directory / "best_agent.gif"
    try:
        print(f"creating GIF from: {checkpoint_path}", flush=True)
        record(checkpoint=checkpoint_path, output=gif_path, seed=GIF_SEED)
    except Exception as error:
        # Checkpoints and metrics are already safely stored at this point.
        print(f"[warn] GIF creation failed: {error}", flush=True)

    return output_directory, random_baseline, final_eval


def _get_experiment_config(seed):
    return {
        "SEED": seed,
        "ENV_TYPE": ENV_TYPE,
        "VEC_OBS_NORMALIZE": "none" if USE_ATTENTION else "near",
        "AGENT_TYPE": "PQN",
        "LR": LR,
        "GAMMA": GAMMA,
        "START_EPS": START_EPS,
        "FINAL_EPS": FINAL_EPS,
        "HIDDEN_SIZE": HIDDEN_SIZE,
        "LAYER_NORM": LAYER_NORM,
        "DUELING_NET": DUELING_NET,
        "USE_ATTENTION": USE_ATTENTION,
        "ATTENTION_NUM_HEADS": ATTENTION_NUM_HEADS,
        "ATTENTION_FUSION": ATTENTION_FUSION,
        "ATTENTION_POSITION_MODE": ATTENTION_POSITION_MODE,
        "DATA_AUGMENTATION": DATA_AUGMENTATION,
        "AUGMENTATION_MODE": AUGMENTATION_MODE,
        "LOSS_FN": LOSS_FN,
        "MAX_GRAD_NORM": MAX_GRAD_NORM,
        "NUM_ENVS": PQN_NUM_ENVS,
        "NUM_STEPS": PQN_NUM_STEPS,
        "TOTAL_TIMESTEPS": PQN_TOTAL_TIMESTEPS,
        "NUM_MINIBATCHES": PQN_NUM_MINIBATCHES,
        "UPDATE_EPOCHS": PQN_UPDATE_EPOCHS,
        "Q_LAMBDA": PQN_Q_LAMBDA,
        "ANNEAL_LR": PQN_ANNEAL_LR,
        "OPTIMIZER": "adam",
        "EXPLORATION_FRACTION": PQN_EXPLORATION_FRACTION,
        "EVAL_EPISODES": EVAL_EPISODES,
        "EVAL_SEED_BASE": EVAL_SEED_BASE,
        "GIF_SEED": GIF_SEED,
        "GIF_FPS": PHYSICS_FPS / N_FRAME_SKIP,
        "GIF_MATCH_MODEL_FRAMES": True,
        "GIF_OBSERVATION_VIEW": "observation[-1], 84x84 grayscale",
        "N_FRAME_SKIP": N_FRAME_SKIP,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Danmaku PQN agent")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--start-eps", type=float, default=START_EPS)
    parser.add_argument("--final-eps", type=float, default=FINAL_EPS)
    parser.add_argument("--exploration-fraction", type=float, default=PQN_EXPLORATION_FRACTION)

    parser.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE)
    parser.add_argument(
        "--layer-norm",
        action=argparse.BooleanOptionalAction,
        default=LAYER_NORM,
        help="target network가 없는 대신 LayerNorm으로 학습을 안정화한다 (PQN은 켜두는 게 기본)",
    )
    parser.add_argument("--dueling-net", action=argparse.BooleanOptionalAction, default=DUELING_NET)
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
        help="minibatch마다 D4 대칭 augmentation 적용",
    )
    parser.add_argument(
        "--augmentation-mode",
        choices=("spawn_safe", "d4"),
        default=AUGMENTATION_MODE,
        help="observation augmentation symmetry group",
    )
    parser.add_argument("--loss-fn", choices=("huber", "mse"), default=LOSS_FN)
    parser.add_argument("--max-grad-norm", type=float, default=MAX_GRAD_NORM)

    parser.add_argument("--num-envs", type=int, default=PQN_NUM_ENVS)
    parser.add_argument("--num-steps", type=int, default=PQN_NUM_STEPS)
    parser.add_argument("--total-timesteps", type=int, default=PQN_TOTAL_TIMESTEPS)
    parser.add_argument("--num-minibatches", type=int, default=PQN_NUM_MINIBATCHES)
    parser.add_argument("--update-epochs", type=int, default=PQN_UPDATE_EPOCHS)
    parser.add_argument("--q-lambda", type=float, default=PQN_Q_LAMBDA)
    parser.add_argument(
        "--anneal-lr",
        action=argparse.BooleanOptionalAction,
        default=PQN_ANNEAL_LR,
    )

    parser.add_argument("--eval-episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--eval-interval", type=int, default=PQN_EVAL_INTERVAL)
    parser.add_argument("--log-interval", type=int, default=PQN_LOG_INTERVAL)
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument("--env-type", choices=("img", "vec"), default=ENV_TYPE)

    return parser.parse_args()


def apply_args(args):
    global SEED, LR, GAMMA, START_EPS, FINAL_EPS
    global HIDDEN_SIZE, LAYER_NORM, DUELING_NET, USE_ATTENTION, ATTENTION_NUM_HEADS
    global ATTENTION_FUSION
    global ATTENTION_POSITION_MODE
    global DATA_AUGMENTATION, AUGMENTATION_MODE, LOSS_FN, MAX_GRAD_NORM
    global PQN_NUM_ENVS, PQN_NUM_STEPS, PQN_TOTAL_TIMESTEPS, PQN_NUM_MINIBATCHES
    global PQN_UPDATE_EPOCHS, PQN_Q_LAMBDA, PQN_EXPLORATION_FRACTION
    global PQN_ANNEAL_LR
    global EVAL_EPISODES, PQN_EVAL_INTERVAL, PQN_LOG_INTERVAL, OUTPUT_ROOT, ENV_TYPE

    SEED = args.seed
    LR = args.lr
    GAMMA = args.gamma
    START_EPS = args.start_eps
    FINAL_EPS = args.final_eps

    HIDDEN_SIZE = args.hidden_size
    LAYER_NORM = args.layer_norm
    DUELING_NET = args.dueling_net
    USE_ATTENTION = args.use_attention
    ATTENTION_NUM_HEADS = args.attention_num_heads
    ATTENTION_FUSION = args.attention_fusion
    ATTENTION_POSITION_MODE = args.attention_position_mode
    DATA_AUGMENTATION = args.data_augmentation
    AUGMENTATION_MODE = args.augmentation_mode
    LOSS_FN = args.loss_fn
    MAX_GRAD_NORM = args.max_grad_norm

    PQN_NUM_ENVS = args.num_envs
    PQN_NUM_STEPS = args.num_steps
    PQN_TOTAL_TIMESTEPS = args.total_timesteps
    PQN_NUM_MINIBATCHES = args.num_minibatches
    PQN_UPDATE_EPOCHS = args.update_epochs
    PQN_Q_LAMBDA = args.q_lambda
    PQN_ANNEAL_LR = args.anneal_lr
    PQN_EXPLORATION_FRACTION = args.exploration_fraction

    EVAL_EPISODES = args.eval_episodes
    PQN_EVAL_INTERVAL = args.eval_interval
    PQN_LOG_INTERVAL = args.log_interval
    OUTPUT_ROOT = args.output_root
    ENV_TYPE = args.env_type


def main():
    args = parse_args()
    apply_args(args)

    output_directory, random_baseline, final_eval = run_experiment(seed=SEED)

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
