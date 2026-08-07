"""학습 결과 저장 (CSV / 그래프 / 체크포인트).

pandas가 설치되어 있지 않으므로 표준 csv 모듈만 쓴다.
matplotlib은 화면 없이 저장만 하므로 Agg 백엔드를 강제한다.
"""

import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np  
import torch 

import config 


def make_experiment_path(agent_type, seed, output_root=None):
    """results/<AGENT_TYPE>_seed<seed>_<timestamp>/ 를 만들어 반환한다."""
    root = Path(output_root if output_root is not None else config.OUTPUT_ROOT)
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    path = root / f"{agent_type}_seed{seed}_{timestamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def save_checkpoint(path, agent, meta):
    """학습 중에도 호출된다 (best 갱신 시)."""
    torch.save(
        {
            "model_state_dict": agent.model.state_dict(),
            "target_model_state_dict": agent.target_model.state_dict(),
            "meta": meta,
        },
        path,
    )


def save_intermediate_results(
    experiment_path,
    config_values,
    episode_records,
    eval_history,
    metric_history,
    agent,
    elapsed_seconds,
):
    """학습 중간 결과 저장"""
    experiment_path = Path(experiment_path)
    experiment_path.mkdir(parents=True, exist_ok=True)

    progress = {
        "config": config_values,
        "elapsed_seconds": elapsed_seconds,
        "episodes_completed": len(episode_records),
        "decision_steps": sum(row["decision_steps"] for row in episode_records),
        "latest_eval": eval_history[-1] if eval_history else None,
        "best_eval": (
            max(eval_history, key=lambda row: row["mean_score"])
            if eval_history else None
        ),
        "latest_metrics": metric_history[-1] if metric_history else None,
        "complete": False,
    }
    (experiment_path / "progress.json").write_text(
        json.dumps(progress, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_csv(
        experiment_path / "episodes.partial.csv",
        episode_records,
        [
            "episode", "decision_steps", "physics_steps", "score", "ep_return",
            "epsilon", "mean_loss", "mean_q",
        ],
    )
    if eval_history:
        _write_csv(
            experiment_path / "eval.partial.csv",
            eval_history,
            [
                "episode", "env_steps", "mean_score", "median_score", "std_score",
                "sem_score", "max_score", "p_ge_120", "p_ge_60", "mean_length",
                "action_entropy",
            ],
        )
    if metric_history:
        _write_csv(
            experiment_path / "metrics.partial.csv",
            metric_history,
            ["episode", "env_steps", "weight_norm", "srank", "srank_max"],
        )
    save_checkpoint(
        experiment_path / "latest.pt",
        agent,
        {
            "episode": len(episode_records),
            "config": config_values,
            "latest_eval": eval_history[-1] if eval_history else None,
            "complete": False,
        },
    )


def _moving_average(values, window):
    if len(values) < 2:
        return np.array(values, dtype=np.float64)
    window = max(1, min(window, len(values)))
    kernel = np.ones(window) / window
    return np.convolve(np.asarray(values, dtype=np.float64), kernel, mode="valid")


def _plot_curves(
    path,
    episode_records,
    eval_history,
    metric_history,
    training_errors,
    random_baseline,
):
    figure = plt.figure(figsize=(14, 16))
    grid = figure.add_gridspec(4, 2)
    axes = np.array([
        [figure.add_subplot(grid[row, column]) for column in range(2)]
        for row in range(3)
    ])

    # 1) 에피소드별 score + 이동평균
    axis = axes[0][0]
    scores = [row["score"] for row in episode_records]
    axis.plot(scores, color="tab:blue", alpha=0.25, linewidth=0.7, label="episode score")
    if len(scores) >= 2:
        window = max(1, len(scores) // 50)
        smoothed = _moving_average(scores, window)
        axis.plot(
            range(len(scores) - len(smoothed), len(scores)),
            smoothed,
            color="tab:blue",
            linewidth=2,
        )
    if random_baseline is not None:
        axis.axhline(
            random_baseline["mean_score"],
            color="gray",
            linestyle="--",
            label=f"random ({random_baseline['mean_score']:.1f})",
        )
    axis.set_xlabel("episode")
    axis.set_ylabel("score")
    axis.set_title("Training score")
    axis.legend(fontsize=8)

    # 2) 평가 곡선 (중앙값)
    axis = axes[0][1]
    if eval_history:
        x = [row["episode"] for row in eval_history]
        median = np.array([row["median_score"] for row in eval_history])
        axis.plot(x, median, marker="o", label="eval median")
        if random_baseline is not None:
            axis.axhline(
                random_baseline["median_score"],
                color="gray",
                linestyle="--",
                label=f"random ({random_baseline['median_score']:.1f})",
            )
        axis.legend(fontsize=8)
    else:
        axis.text(0.5, 0.5, "no eval", ha="center", va="center")
    axis.set_xlabel("episode")
    axis.set_ylabel("score")
    axis.set_title("Greedy evaluation")

    # 3) 학습 loss
    axis = axes[1][0]
    if training_errors:
        axis.plot(training_errors, color="tab:blue", alpha=0.3, linewidth=0.7)
        if len(training_errors) >= 2:
            window = max(1, len(training_errors) // 200)
            smoothed = _moving_average(training_errors, window)
            axis.plot(
                range(len(training_errors) - len(smoothed), len(training_errors)),
                smoothed,
                color="tab:blue",
                linewidth=2,
            )
        axis.set_yscale("log")
    else:
        axis.text(0.5, 0.5, "no gradient step", ha="center", va="center")
    axis.set_xlabel("gradient step")
    axis.set_ylabel("loss")
    axis.set_title("TD loss")

    # 4) 행동 엔트로피 (정책 붕괴 감시)
    axis = axes[1][1]
    if eval_history:
        axis.plot(
            [row["episode"] for row in eval_history],
            [row["action_entropy"] for row in eval_history],
            color="tab:orange",
            marker="s",
            markersize=3,
        )
        axis.axhline(np.log(9), color="tab:orange", linestyle=":", alpha=0.5)
    axis.set_xlabel("episode")
    axis.set_ylabel("action entropy (nats), max=ln9=2.197")
    axis.set_title("Policy collapse check")

    # 5) online Q-network weight norm
    axis = axes[2][0]
    if metric_history:
        axis.plot(
            [row["episode"] for row in metric_history],
            [row["weight_norm"] for row in metric_history],
        )
    else:
        axis.text(0.5, 0.5, "no metric", ha="center", va="center")
    axis.set_xlabel("episode")
    axis.set_ylabel("L2 norm")
    axis.set_title("Weight norm")

    # 6) Kumar et al. singular-value threshold rank
    axis = axes[2][1]
    valid_rank_rows = [
        row for row in metric_history
        if row["srank"] != "" and np.isfinite(row["srank"])
    ]
    if valid_rank_rows:
        axis.plot(
            [row["episode"] for row in valid_rank_rows],
            [row["srank"] for row in valid_rank_rows],
        )
    else:
        axis.text(0.5, 0.5, "waiting for 512 replay samples", ha="center", va="center")
    axis.set_xlabel("episode")
    axis.set_ylabel("srank")
    axis.set_title("Representation srank")

    # 7) 에피소드별 online Q-network의 평균 Q(s, a)
    axis = figure.add_subplot(grid[3, :])
    valid_q_rows = [
        row for row in episode_records
        if row.get("mean_q", "") != "" and np.isfinite(row["mean_q"])
    ]
    if valid_q_rows:
        q_episodes = [row["episode"] for row in valid_q_rows]
        mean_q_values = [row["mean_q"] for row in valid_q_rows]
        axis.plot(
            q_episodes,
            mean_q_values,
            color="tab:blue",
            alpha=0.3,
            linewidth=0.7,
            label="episode mean Q",
        )
        if len(mean_q_values) >= 2:
            window = max(1, len(mean_q_values) // 50)
            smoothed = _moving_average(mean_q_values, window)
            axis.plot(
                q_episodes[len(mean_q_values) - len(smoothed):],
                smoothed,
                color="tab:blue",
                linewidth=2,
            )
        axis.legend(fontsize=8)
    else:
        axis.text(0.5, 0.5, "no gradient step", ha="center", va="center")
    axis.set_xlabel("episode")
    axis.set_ylabel("mean Q value")
    axis.set_title("Mean Q value")

    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def save_experiment_results(
    experiment_path,
    config_values,
    episode_records,
    eval_history,
    metric_history,
    training_errors,
    random_baseline,
    final_eval,
    agent,
    elapsed_seconds,
):
    experiment_path = Path(experiment_path)
    experiment_path.mkdir(parents=True, exist_ok=True)

    summary = {
        "config": config_values,
        "elapsed_seconds": elapsed_seconds,
        "episodes_completed": len(episode_records),
        "decision_steps": sum(row["decision_steps"] for row in episode_records),
        "gradient_steps": len(training_errors),
        "random_baseline": random_baseline,
        "final_eval": final_eval,
        "best_eval": max(eval_history, key=lambda row: row["mean_score"]) if eval_history else None,
        "latest_metrics": metric_history[-1] if metric_history else None,
    }
    (experiment_path / "results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (experiment_path / "config.json").write_text(
        json.dumps(config_values, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    _write_csv(
        experiment_path / "episodes.csv",
        episode_records,
        [
            "episode", "decision_steps", "physics_steps", "score", "ep_return",
            "epsilon", "mean_loss", "mean_q",
        ],
    )

    if eval_history:
        _write_csv(
            experiment_path / "eval.csv",
            eval_history,
            [
                "episode", "env_steps", "mean_score", "median_score", "std_score",
                "sem_score", "max_score", "p_ge_120", "p_ge_60", "mean_length",
                "action_entropy",
            ],
        )

    if metric_history:
        _write_csv(
            experiment_path / "metrics.csv",
            metric_history,
            ["episode", "env_steps", "weight_norm", "srank", "srank_max"],
        )

    # loss는 개수가 많으므로 최대 20000개로 subsample
    if training_errors:
        stride = max(1, len(training_errors) // 20000)
        _write_csv(
            experiment_path / "training_error.csv",
            [
                {"gradient_step": index, "loss": value}
                for index, value in enumerate(training_errors)
                if index % stride == 0
            ],
            ["gradient_step", "loss"],
        )

    save_checkpoint(
        experiment_path / "last.pt",
        agent,
        {"config": config_values, "final_eval": final_eval},
    )

    _plot_curves(
        experiment_path / "curves.png",
        episode_records,
        eval_history,
        metric_history,
        training_errors,
        random_baseline,
    )

    # 정상 종료 시에는 크래시 복구용 중간 산출물(partial csv, progress.json,
    # latest.pt)이 최종본(csv, results.json, last.pt)과 완전히 중복되므로 제거한다.
    for stale_name in (
        "episodes.partial.csv",
        "eval.partial.csv",
        "metrics.partial.csv",
        "progress.json",
        "latest.pt",
    ):
        stale_path = experiment_path / stale_name
        if stale_path.exists():
            stale_path.unlink()

    return experiment_path
