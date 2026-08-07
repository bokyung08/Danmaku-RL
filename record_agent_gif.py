import argparse
import json
from pathlib import Path

import torch
from PIL import Image

import config
from env import DanmakuImgEnv, DanmakuVecEnv
from model import NatureCNN, MLP, AttentionQNetwork
from render import Renderer


def _capture_frame(renderer, game):
    # human.py가 플레이할 때 보는 것과 동일한 렌더 (agent, ball, score).
    return Image.fromarray(renderer.get_image(game, view_score=True)).convert("RGB")


def _quantize_frame(frame):
    return frame.quantize(
        colors=96,
        method=Image.Quantize.MEDIANCUT,
        dither=Image.Dither.NONE,
    )


def _model_frame_durations(frame_count, fps):
    """Represent the model cadence using GIF's 10 ms time resolution."""
    exact_ms = 1000.0 / fps
    boundaries = [round(index * exact_ms / 10.0) * 10 for index in range(frame_count + 1)]
    durations = [
        max(10, boundaries[index + 1] - boundaries[index])
        for index in range(frame_count)
    ]
    durations[-1] = 1200
    return durations


def _is_image_checkpoint(state_dict):
    """conv.* 파라미터가 있으면 NatureCNN(이미지) 체크포인트, 없으면 MLP(벡터) 체크포인트다."""
    return any(key.startswith("conv.") for key in state_dict)


def record(checkpoint, output, seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    metadata = payload.get("meta", {})

    # 학습때 사용한 config 불러오기
    saved_config = {}
    config_path = checkpoint.parent / "config.json"
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as file:
            saved_config.update(json.load(file))
    saved_config.update(metadata.get("config", {}))

    hidden_size = int(saved_config.get("HIDDEN_SIZE", config.HIDDEN_SIZE))
    layer_norm = bool(saved_config.get("LAYER_NORM", False))
    dueling_net = bool(saved_config.get("DUELING_NET", False))
    use_attention = bool(saved_config.get("USE_ATTENTION", False))
    attention_num_heads = int(saved_config.get("ATTENTION_NUM_HEADS", config.ATTENTION_NUM_HEADS))
    state_dict = payload["model_state_dict"]
    inferred_fusion = "concat" if any(key.startswith("fuse.") for key in state_dict) else "residual"
    attention_fusion = str(saved_config.get("ATTENTION_FUSION", inferred_fusion))
    attention_position_mode = str(
        saved_config.get("ATTENTION_POSITION_MODE", config.ATTENTION_POSITION_MODE)
    )
    frame_skip = int(saved_config.get("N_FRAME_SKIP", config.N_FRAME_SKIP))
    if frame_skip <= 0:
        raise ValueError("N_FRAME_SKIP must be positive")

    is_image_model = _is_image_checkpoint(payload["model_state_dict"])
    if is_image_model:
        model_cls, extra_kwargs = NatureCNN, {}
    elif use_attention:
        model_cls, extra_kwargs = AttentionQNetwork, {
            "num_heads": attention_num_heads,
            "fusion_mode": attention_fusion,
            "position_mode": attention_position_mode,
        }
    else:
        model_cls, extra_kwargs = MLP, {}

    # env.step()은 config.N_FRAME_SKIP을 그대로 참조한다. 학습 당시 값과 현재 전역
    # 설정이 다를 수 있으므로 체크포인트에 저장된 값을 녹화하는 동안만 적용한다.
    original_frame_skip = config.N_FRAME_SKIP
    config.N_FRAME_SKIP = frame_skip
    try:
        env = (
            DanmakuImgEnv()
            if is_image_model
            else DanmakuVecEnv(
                normalize=saved_config.get(
                    "VEC_OBS_NORMALIZE", "none" if use_attention else "near"
                )
            )
        )
        # imgenv는 존재하는 renderer 사용
        renderer = env.renderer if is_image_model else Renderer(render_mode="rgb_array")

        model = model_cls(
            env.observation_shape,
            env.n_actions,
            hidden_size=hidden_size,
            layer_norm=layer_norm,
            dueling_net=dueling_net,
            **extra_kwargs,
        ).to(device)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()

        observation, _ = env.reset(seed=seed)
        frames = []
        observation_frames = []
        # One GIF frame corresponds to one observation/action decision.
        effective_fps = config.PHYSICS_FPS / frame_skip
        done = False

        while not done:
            with torch.inference_mode():
                tensor = torch.as_tensor(observation, device=device).unsqueeze(0)
                last_action = int(model(tensor).argmax(dim=1).item())

            # Capture the exact state from which the model chose this action,
            # before env.step() advances it.
            if is_image_model:
                observation_frames.append(Image.fromarray(observation[-1].copy()))
            frames.append(_quantize_frame(_capture_frame(renderer, env.game)))

            observation, _, terminated, truncated, _ = env.step(last_action)
            done = terminated or truncated

        if not frames:
            raise RuntimeError("녹화된 프레임이 없습니다.")

        # Keep the terminal state as an extra presentation frame.
        frames.append(_quantize_frame(_capture_frame(renderer, env.game)))
        if is_image_model:
            observation_frames.append(Image.fromarray(observation[-1].copy()))

        durations = _model_frame_durations(len(frames), effective_fps)
        output.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            output,
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=0,
            optimize=True,
            disposal=2,
            comment=f"checkpoint={checkpoint.name};seed={seed};score={env.game.state.score}".encode(),
        )

        # 이미지 모델일땐 (84, 84) 도 뽑음
        observation_output = None
        if is_image_model:
            observation_output = output.with_name(f"{output.stem}_observation{output.suffix}")
            observation_frames[0].save(
                observation_output,
                save_all=True,
                append_images=observation_frames[1:],
                duration=durations,
                loop=0,
                optimize=False,
                disposal=2,
                comment=(
                    f"checkpoint={checkpoint.name};seed={seed};score={env.game.state.score};"
                    "view=observation[-1];size=84x84;mode=grayscale"
                ).encode(),
            )

        env.close()
        if not is_image_model:
            renderer.close()
    finally:
        config.N_FRAME_SKIP = original_frame_skip

    print(f"saved={output}")
    if observation_output is not None:
        print(f"saved_observation={observation_output}")
    else:
        print("saved_observation=skipped (vector-observation model has no 84x84 frame to show)")
    print(
        f"seed={seed} score={env.game.state.score} frames={len(frames)} "
        f"fps={effective_fps:.2f} device={device} model={'image' if is_image_model else 'vector'}"
    )
    return output


def main():
    parser = argparse.ArgumentParser(description="학습된 Danmaku 체크포인트를 GIF로 녹화합니다.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=20012)
    args = parser.parse_args()
    record(
        args.checkpoint,
        args.output,
        args.seed,
    )


if __name__ == "__main__":
    main()
