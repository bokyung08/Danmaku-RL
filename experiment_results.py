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
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import config  # noqa: E402


TARGET_SCORE = 120


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


def _moving_average(values, window):
    if len(values) < 2:
        return np.array(values, dtype=np.float64)
    window = max(1, min(window, len(values)))
    kernel = np.ones(window) / window
    return np.convolve(np.asarray(values, dtype=np.float64), kernel, mode="valid")


def _plot_curves(path, episode_records, eval_history, training_errors, random_baseline):
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))

    # 1) 에피소드별 score + 이동평균
    axis = axes[0][0]
    scores = [row["score"] for row in episode_records]
    axis.plot(scores, alpha=0.25, linewidth=0.7, label="episode score")
    if len(scores) >= 2:
        window = max(1, len(scores) // 50)
        smoothed = _moving_average(scores, window)
        axis.plot(
            range(len(scores) - len(smoothed), len(scores)),
            smoothed,
            linewidth=2,
            label=f"moving avg ({window})",
        )
    if random_baseline is not None:
        axis.axhline(
            random_baseline["mean_score"],
            color="gray",
            linestyle="--",
            label=f"random ({random_baseline['mean_score']:.1f})",
        )
    axis.axhline(TARGET_SCORE, color="red", linestyle=":", label=f"target ({TARGET_SCORE})")
    axis.set_xlabel("episode")
    axis.set_ylabel("score")
    axis.set_title("Training score")
    axis.legend(fontsize=8)

    # 2) 평가 곡선 (평균 +- SEM)
    axis = axes[0][1]
    if eval_history:
        x = [row["episode"] for row in eval_history]
        mean = np.array([row["mean_score"] for row in eval_history])
        sem = np.array([row["sem_score"] for row in eval_history])
        axis.plot(x, mean, marker="o", label="eval mean")
        axis.fill_between(x, mean - sem, mean + sem, alpha=0.25, label="+-SEM")
        if random_baseline is not None:
            axis.axhline(
                random_baseline["mean_score"],
                color="gray",
                linestyle="--",
                label=f"random ({random_baseline['mean_score']:.1f})",
            )
        axis.axhline(TARGET_SCORE, color="red", linestyle=":", label=f"target ({TARGET_SCORE})")
        axis.legend(fontsize=8)
    else:
        axis.text(0.5, 0.5, "no eval", ha="center", va="center")
    axis.set_xlabel("episode")
    axis.set_ylabel("score")
    axis.set_title("Greedy evaluation")

    # 3) 학습 loss
    axis = axes[1][0]
    if training_errors:
        axis.plot(training_errors, alpha=0.3, linewidth=0.7)
        if len(training_errors) >= 2:
            window = max(1, len(training_errors) // 200)
            smoothed = _moving_average(training_errors, window)
            axis.plot(
                range(len(training_errors) - len(smoothed), len(training_errors)),
                smoothed,
                linewidth=2,
            )
        axis.set_yscale("log")
    else:
        axis.text(0.5, 0.5, "no gradient step", ha="center", va="center")
    axis.set_xlabel("gradient step")
    axis.set_ylabel("loss")
    axis.set_title("TD loss")

    # 4) epsilon + 행동 엔트로피 (정책 붕괴 감시)
    axis = axes[1][1]
    axis.plot([row["epsilon"] for row in episode_records], label="epsilon")
    axis.set_xlabel("episode")
    axis.set_ylabel("epsilon")
    if eval_history:
        twin = axis.twinx()
        twin.plot(
            [row["episode"] for row in eval_history],
            [row["action_entropy"] for row in eval_history],
            color="tab:orange",
            marker="s",
            markersize=3,
            label="greedy action entropy",
        )
        twin.axhline(np.log(9), color="tab:orange", linestyle=":", alpha=0.5)
        twin.set_ylabel("action entropy (nats), max=ln9=2.197")
    axis.set_title("Exploration / policy collapse")
    axis.legend(fontsize=8, loc="upper right")

    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def save_experiment_results(
    experiment_path,
    config_values,
    episode_records,
    eval_history,
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
        ["episode", "decision_steps", "physics_steps", "score", "ep_return", "epsilon", "mean_loss"],
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
        training_errors,
        random_baseline,
    )

    return experiment_path
