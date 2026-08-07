# Danmaku-RL

탄막 회피 게임을 강화학습으로 푸는 프로젝트. 600x600 화면에서 3초마다 좌상단에 공이
하나씩 생기고(최대 40개), 벽에 반사되며 날아다니는 공을 피해 오래 살아남으면 된다.
점수 = 생존 초 수. 180초(=10800 프레임)에서 종료하므로 만점은 180점.

- 행동: 9개 (정지 + 8방향)

## 데모

최종 학습된 에이전트(`final_results/PQN_seed1_0805_185626_best/`). 왼쪽은 실제 플레이,
오른쪽은 같은 모델의 attention을 공 색으로 칠한 것(빨강에 가까울 수록 주목한다는 뜻).

<table>
<tr>
<td width="50%"><img src="final_results/PQN_seed1_0805_185626_best/best_agent.gif" width="100%"></td>
<td width="50%"><video src="final_results/PQN_seed1_0805_185626_best/attention_heatmap_seed30000.mp4" width="100%" controls muted loop></video></td>
</tr>
<tr>
<td align="center"><a href="final_results/PQN_seed1_0805_185626_best/best_agent.gif">best_agent.gif</a> (seed 20000)</td>
<td align="center"><a href="final_results/PQN_seed1_0805_185626_best/attention_heatmap_seed30000.mp4">attention_heatmap_seed30000.mp4</a> (seed 30000)</td>
</tr>
</table>

head별로 나눠 본 영상은
[`attention_heatmap_seed30000_heads.mp4`](final_results/PQN_seed1_0805_185626_best/attention_heatmap_seed30000_heads.mp4).

## 실행

```bash
pip install -r requirements.txt

python human.py                              # 직접 플레이 (방향키)
python train.py                              # DQN / DDQN 학습
python train_pqn.py                          # PQN 학습
python evaluate_checkpoint.py <best.pt>      # 체크포인트 평가
python record_agent_gif.py <best.pt> out.gif # 플레이 GIF 녹화
python analyze_attention.py <best.pt>        # attention 히트맵 mp4
```

하이퍼파라미터는 `config.py`에 있고, 대부분 CLI 인자로도 덮어쓸 수 있다.
결과는 `results/<AGENT>_seed<N>_<시각>/`에 CSV / 그래프 / 체크포인트로 저장된다.

## 접근

### 환경 (`env.py`)

| | 관측 | 모델 |
|---|---|---|
| `DanmakuImgEnv` | 84x84 흑백 4프레임 스택 | NatureCNN |
| `DanmakuVecEnv` | 2 + 5x40 = 202차원 벡터 | MLP / Attention |

벡터 관측은 `[agent x,y] + [ball x,y,vx,vy,mask] x 40`. 공이 40개 미만이면 mask=0으로
패딩한다. 둘 다 frame skip 4 (=15Hz로 행동 결정).

### 모델 (`model.py`)

**AttentionQNetwork** — agent를 query, 공 40개를 key/value로 두는 cross-attention.
공마다 같은 encoder를 공유하므로 공 순서에 무관하고, mask로 빈 슬롯을 제외한다.
모든 공이 빈 슬롯이면 softmax가 NaN이 되므로 마스킹되지 않는 학습 파라미터
(`empty_kv`)를 하나 붙였다.

MLP는 같은 벡터를 그대로 flatten해서 쓰되, 공을 거리순으로 정렬해서 넣는다.

### 알고리즘 (`agent.py`)

- **DQN / DDQN** — replay buffer + target network
- **PQN** — replay buffer와 target network 없이, 128개 환경을 동시에 굴려 모은
  rollout으로 Q(lambda) return을 계산해 학습. LayerNorm이 안정화를 담당한다.
  ([cleanrl 구현](https://docs.cleanrl.dev/rl-algorithms/pqn/) 참고)

### Data augmentation (`data_augmentation.py`)

공이 항상 좌상단에서만 생성되므로 D4 대칭 8개를 다 쓰면 실제로 존재하지 않는 상태를
학습하게 된다. 스폰 지점과 속도 분포를 보존하는 전치 `(x,y) -> (y,x)`만 남겨
`{항등, 전치}` 2개를 쓴다 (`spawn_safe`). 관측과 함께 행동도 같이 변환한다.

## 결과

전체 실험은 `final_results/`에 있고, 학습 곡선 비교는 `development.ipynb` 참고.

- `학습` = 마지막 100 에피소드 평균 점수 (탐색 포함)
- `평가` = 고정 시드 20개에 대한 greedy 평가 중 최고 (`results.json`의 `best_eval`)
- random 정책 baseline = 9.8

비교 축 별로 묶어서 정렬했다.

| 런 (`final_results/`) | 알고리즘 | 관측 / 모델 | 설정 | 학습량 | 시간 | 학습 | 평가 |
|---|---|---|---|---|---|---|---|
| **관측·모델 비교 (DDQN, 3000 ep)** | | | | | | | |
| `ddqn_img_cnn` | DDQN | img / CNN | - | 3000 ep | 16분 | 9.2 | 15.5 |
| `ddqn_vec_mlp` | DDQN | vec / MLP | - | 3000 ep | 15분 | 15.2 | 17.6 |
| `ddqn_vec_attention` | DDQN | vec / Attention | LN | 3000 ep | 75분 | 37.9 | 49.4 |
| **더 오래 학습하면 되는가** | | | | | | | |
| `ddqn_img_cnn_aug_5000` | DDQN | img / CNN | DA | 5000 ep | 38분 | 11.2 | 15.4 |
| `ddqn_vec_mlp_10000` | DDQN | vec / MLP | - | 10000 ep | 74분 | 21.4 | 26.8 |
| `pqn_img_cnn_aug_20m` | PQN | img / CNN | DA | 20M step | 139분 | 25.9 | 27.0 |
| **PQN 튜닝 (예산 고정, 1M step)** | | | | | | | |
| `pqn_vec_attention_1m_epoch2` | PQN | vec / Attention | anneal | 1M step | 5.0분 | 54.9 | 57.8 |
| `pqn_vec_attention_1m_epoch2_aug` | PQN | vec / Attention | anneal + DA | 1M step | 4.7분 | 63.0 | 66.5 |
| `pqn_vec_attention_1m_epoch4` | PQN | vec / Attention | anneal + epoch 4 | 1M step | 5.6분 | 61.7 | 77.1 |
| `pqn_vec_attention_1m_epoch4_aug` | PQN | vec / Attention | anneal + DA + epoch 4 | 1M step | 6.4분 | 70.3 | 83.1 |
| **예산 확대** | | | | | | | |
| `pqn_vec_attention` | PQN | vec / Attention | - | 5M step | 29분 | 62.1 | 83.1 |
| `pqn_vec_attention_anneal` | PQN | vec / Attention | anneal | 5M step | 32분 | 99.9 | 109.4 |
| **최종** | | | | | | | |
| `PQN_seed1_0805_185626_best` | PQN | vec / Attention | anneal + DA + epoch 4 | 100M step | 954분 | **144.2** | **175.5** |

최종 체크포인트는 평균 175.5 / 중앙값 180 / P(>=120) = 1.00 으로 목표 120점을 넘겼다.

정리하면

1. **이미지 관측은 실패했다.** 학습에 더 많은 이미지를 병렬 env로 처리한다면 학습이 가능할 것이라고 생각 중 (development.ipynb 마지막 셀 참고)
2. **관측 구조가 알고리즘보다 크게 작용했다.** 같은 DDQN에서 MLP 15.2 -> Attention 37.9.
3. **PQN이 DQN 계열보다 잘 됐다.** 병렬 환경으로 데이터를 훨씬 빨리 모으고,
   Q(lambda)가 죽는 순간의 페널티를 한 번의 업데이트로 앞쪽 스텝까지 전파한다.
4. augmentation, LR annealing, update epoch 4는 각각 조금씩 도움이 됐다.

학습된 정책의 행동 분포를 보면 공의 진행 방향(우하단)과 정반대인 up-left가 3.2%로 가장
적고, 수직으로 피하는 up-right / down-left가 15.9% / 14.0%로 가장 많다. 전치 대칭으로
짝지어지는 행동끼리 사용률이 거의 같아서 augmentation이 의도대로 작동했음을 보여준다.

`analyze_attention.py`로 뽑은 통계에서도 충돌까지 4프레임 이하로 남은 상황에서는
attention 1순위가 실제 위험한 공을 가리킬 확률이 98.7% (무작위 기준 3.0%)인 반면,
안전한 상황에서는 7.3% (무작위 8.4%)로 사실상 아무것도 보지 않는다.

## 한계

- PQN은 병렬로 env를 처리하지만, 현재 env는 병렬 구조를 지원하지 않아, for문을 이용해서 처리함. 따라서 PQN으로 오는 속도 차이는 병렬처리라기 보단, replay buffer가 없고 sample을 cpu, gpu로 옮기는 작업이 현저히 적기 때문으로 파악 
- 이미지가 더 잘 학습할 것이라는 처음 기대완 다르게 vecenv보다 현저히 느린 학습을 보여줌. 공의 속도가 너무 빠르거나 벽에 반사와 같은 로직을 학습하기엔 데이터가 너무 적은게 아닌가라는 추측
- Data augmentation은 두가지 방법 중 spawn-safe 방식만 사용했지만, 수정한다면 D4만큼 augment를 진행하면서 mdp가 안전하게 만들 수 있지 않을까 추측 (e.g. 공이 생성되는 경우만 spawn-safe 방식 진행. 나머지는 D4)
- PPO 와 같은 actor-critic 방법을 시도해보지 않음. 
- DQN-based는 episode 단위, PQN은 step 단위로 logging을 진행해서 해석에 어려움이 있는 부분이 존재. 
- replay buffer는 imgenv에서 obs, next_obs를 따로 저장. 현재 3프레임이 겹치므로, 더 save할 수 있는 방법 (lazy frames) 존재. 이러면 capacity=1,000,000으로 가도 램이 부족하지 않을듯


## 파일

```
game.py entity.py       게임 로직 (물리, 충돌)
render.py human.py      렌더링 / 직접 플레이
env.py                  RL 환경 (img, vec)
model.py                NatureCNN, MLP, AttentionQNetwork
agent.py buffer.py      DQN, DDQN, PQN, replay buffer, evaluate()
data_augmentation.py    대칭 augmentation
train.py train_pqn.py   학습 스크립트
metric.py               weight norm, srank
experiment_results.py   CSV / 그래프 / 체크포인트 저장
evaluate_checkpoint.py  체크포인트 평가
record_agent_gif.py     플레이 GIF 녹화
analyze_attention.py    attention 히트맵 영상
development.ipynb       실험 결과 비교
```
