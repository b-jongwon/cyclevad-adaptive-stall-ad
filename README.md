# CycleVAD Adaptive Stall AD · 정상 공정 영상 이상 탐지

팀원 [CycleVAD](https://github.com/PigeonLabs/KNU_Capstone1_CycleVAD)를 기반으로 수행한 **전체 16개 장면 정상-only 경보/점수 개선의 탐색 연구**다. 원본 출처는 [ATTRIBUTION](docs/ATTRIBUTION.md)에 구분한다. 전체 모델을 새로 발명한 프로젝트가 아니다.

> **현재 상태:** 학습·실험 결과와 실행 코드는 보존했으나 독립 확인 실험은 미완료다. motion 시작부의 미래 참조가 확인되어 완전 온라인 모델이라고 주장하지 않는다. 테스트 비교 후 선택한 후보이며 모든 장면의 모든 지표 개선을 보장하지 않는다.

## 상세 문서 읽는 순서

- [기초·데이터·지표](docs/review/01_foundations.md)
- [팀원 구성 요소 A to Z](docs/review/02_team_components.md)
- [팀원 A/C/P/CP/Full·E0–E17](docs/review/03_team_experiments.md)
- [팀원 E11/E11B/E12 통합·phase 공간 비교 수치](docs/review/03b_integrated_evidence.md)
- [초기 경보 실험과 전체16 공통 규칙](docs/review/04_local_alarm.md)
- [AE/GRU와 memory 직접 학습](docs/review/05_feature_learning.md)
- [최종 정지 연속 score 결합](docs/review/06_final_fusion.md)
- [모든54설정·전체장면·팀원 비교](docs/review/07_results.md)
- [논문 근거·재현·미검증 과제](docs/review/08_paper_reproduction.md)

## 프로젝트 요약


팀원이 공개한 **CycleVAD**와 그 코드를 기반으로 로컬 GPU에서 수행한 **적응형 임계값·정지 점수·특징 head 비교 실험**을 구분해서 설명한다. 과거 LLM/VLM 프로젝트 전체의 회고 문서는 아니다. 이번 CycleVAD 계열에는 GPT, VLM의 언어 추론, GroundingDINO, 객체 추적이 들어가지 않는다.

팀원 원본: https://github.com/PigeonLabs/KNU_Capstone1_CycleVAD

로컬 확장 공개 저장소: https://github.com/b-jongwon/cyclevad-adaptive-stall-ad

## 먼저 알아야 할 결론

팀원의 방법은 영상 속 공정의 진행 위치를 숫자로 추적하고, 그 위치에서 예상되는 모습과 실제 모습을 비교하는 정상 학습 기반 이상 탐지다. 로컬 개선은 여기에 **영상별 점수 수준 차이를 일부 보정하는 경보 규칙**과 **평소보다 움직임이 부족한 상태를 측정하는 정지 점수**를 추가했다.

추가로 오토인코더, GRU, 최근접 정상 메모리도 실제로 학습·비교했지만, 이 실험에서는 평균 AUROC를 개선하지 못해 최종 균형 후보에서 제외했다. 복잡한 모듈을 많이 넣는 것이 곧 성능 향상을 뜻하지 않는다.

## 문서 지도

1. **기초·데이터·평가:** 정상 학습, 영상/프레임, 분할, 점수와 경보, 지표를 설명한다.
2. **팀원 CycleVAD 구성 요소:** DINOv2, cycle tracker, Fourier 평균, subspace, A/C/P를 하나씩 설명한다.
3. **팀원 실험과 파이프라인:** A, C, P, CP, Full 및 E0–E17 실험을 해석한다.
4. **로컬 경보 개선:** 초기 4개 장면 탐색과 전체 16개 장면 공통 규칙을 구분한다.
5. **직접 학습한 모델:** 정상 memory, AE, GRU, 결합 실험과 학습량을 설명한다.
6. **최종 정지 점수 결합:** 선택 후보의 학습·추론·수식·한계를 설명한다.
7. **모든 결과와 비교 조건:** 주요 평균, 장면별 결과, 54개 설정, 팀원 수치를 제시한다.
8. **논문·재현·검증 과제:** 주장할 수 있는 기여와 아직 확인하지 못한 것을 분리한다.

## 파이프라인 계보

```text
팀원 원본 CycleVAD
  정상 영상 → 동결 DINOv2 → 연속 cycle 위치 θ
  → A: 전체 외형 / C: 위치 조건부 외형 / P: 진행 이상
  → 정상 reference로 점수 보정 → 정상 holdout으로 임계값 → 경보

로컬 초기 4장면 실험
  기존 점수 유지 + 영상 강도 적응형 임계값 + 별도 정지 경보

로컬 전체 16장면 공통 경보
  기존 점수 유지 + 정상 validation에서 선택한 동일 경보 규칙

로컬 특징 학습 비교
  DINOv2 특징 → 정상 memory / AE / GRU → 여러 연속 점수
  → 기존 공통 경보와 동일한 조건으로 평가

로컬 최종 균형 후보
  기존 연속 점수 + 정상 위치별 움직임 부족 점수
  → max 결합 → 새 정상 임계값 → 적응형 경보 + 정지 경보
```

## 실험의 소유와 의미

기본 특징 추출·CycleTracker·CycleVAD·팀원 E0–E17 결과는 팀원 저장소에서 가져온 구성이다. 로컬 확장의 핵심 코드는 `adaptive_final.py`, `universal_alarm.py`, `feature_study.py`, `ranking_fusion.py`다. 팀원의 기존 알고리즘 전체를 우리가 새로 발명했다고 쓰면 안 된다.

최종 후보는 전체 16개 장면 평균에서 기존 로컬 모델보다 AUROC/AP와 주요 경보 지표가 개선되었다. 하지만 R04·S10의 AUROC는 하락했고, 모든 장면의 모든 지표가 개선된 것은 아니다. 실제 테스트 결과를 보고 최종 결합 후보를 골랐으므로 **탐색 결과**이며 독립 확인 실험이 필요하다.

또한 이번 문서 검토 중 정지 점수의 시작 부분에서 미래 관측을 사용하는 초기화가 확인되었다. 기존 수치를 바꾸지 않고 공개하되, 최종 로컬 모델의 완전한 인과적 온라인 동작이 검증되었다고 주장하지 않는다. 상세 내용은 6·8페이지에 있다.

| 설정 | AUROC | AP | FPR ↓ | Recall | Precision | F1 | Balanced acc. | Event coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 초기 고정 경보 | 67.28% | 41.37% | 10.12% | 17.24% | 30.53% | 18.58% | 53.56% | 43.24% |
| 직전 공통 경보 | 67.28% | 41.37% | 9.40% | 30.83% | 54.65% | 35.14% | 60.72% | 80.27% |
| 최종 max_mix_0.1 | 73.41% | 54.73% | 8.58% | 32.08% | 56.85% | 37.06% | 61.75% | 84.21% |

## 회의 때 한 문장으로 설명하기

“팀원 CycleVAD를 기반으로, 정상 영상마다 점수 수준이 달라 생기는 오경보와 움직임 부족 이상을 보완했다. 전체 16개 장면에서 같은 규칙을 사용했고, 신경망을 더 붙이는 방법보다 정지 정보를 점수와 경보에 반영하는 쪽이 이 탐색 실험에서는 더 나았다.”


## 최종 후보의 학습과 추론



1. 정상 fit에서 기존 CycleVAD 템플릿·Fourier 평균·subspace·local memory를 학습한다.
2. 정상 reference에서 기존 A/C/P의 tail 보정을 추정한다.
3. 정상 threshold에서 기존 최종 score의 q99 임계값 Tbase를 구한다.
4. 정상 reference에서 lag1/4 motion의 θ-bin별 lower-tail 분포를 추정한다.
5. min 결합+EMA 0.5를 거친 stall score의 q99.5 임계값 Tstall을 정상 threshold에서 구한다.
6. 두 score를 각 정상 기준으로 나누고 후보 결합을 구성한다.
7. 결합 score의 정상 threshold q99를 다시 구한다. 무조건 임계값 1로 고정하지 않는다.
8. 정상 reference로 결합 score의 시작부 기준 강도를 구한다.
9. 동일한 gain 0.3·persistence 1·연속 stall 3관측 정책으로 최종 경보를 만든다.

## 최종 후보 max_mix_0.1

```text
b(t) = 기존 CycleVAD score(t) / Tbase
u(t) = stall score(t) / Tstall
f(t) = max(b(t), 1.1 × u(t))

Tf = 정상 threshold 자료에서 f의 q99
Δ = max(0, 테스트 처음24관측의 낮은20% 중앙값 − 정상 reference 기준)
Tdynamic(t) = Tf                       # 처음23관측
              Tf + 0.3 × Δ             # 24번째부터

final_alarm(t) = [f(t) > Tdynamic(t)]
                OR [원래 stall score > Tstall가 연속3관측]
```

두 분기를 정상 경보 기준으로 나눠 단위를 맞춘다. b=1이면 기존 정상 경보 기준과 같은 강도다. u=1이면 정상 정지 기준과 같은 강도다. 이는 정상 기준에 대한 상대 강도이며 불량 확률이 아니다.

`max_mix_0.1`의 0.1은 가중 평균에서 정지 비중 10%라는 뜻이 아니다. 코드에서는 `max(b, (1+0.1)×u)`이다. 정지 분기의 값을 1.1배한 뒤 더 큰 분기를 읽는다. 평균 score에서는 강한 정지 증거가 낮은 외형 score에 희석될 수 있어 max 후보를 비교했다.

## 추론 순서

```text
테스트 프레임 → 동결 DINOv2 → global/patch 특징
  ├ 기존 CycleVAD → θ, A/C/P → base score
  └ global cosine lag1/4 → θ별 motion lower-tail → min → EMA → stall score
→ 정상 강도로 normalize → max 결합
→ 해당 결합의 정상 q99 + 초기 강도 보정 → 외형/결합 경보
→ 별도 연속 정지 경보 OR
→ score·alarm·dynamic_threshold·원본 frame indices 출력
→ 그 뒤에 정답 라벨로 성능 계산
```

`balanced_pipeline.joblib`은 base model, motion calibrator, duration limit, 각 후보 threshold/baseline, selected method, alarm parameters를 함께 담는다. `predict_fusion(data, bundle)`은 라벨을 받지 않고 이 bundle로 계산한다. duration limit은 여러 비교 후보를 지원하려고 bundle에 들어 있지만 최종 max_mix_0.1은 duration penalty를 사용하지 않는다.



## 직접 추가 학습한 것

16개 장면별 AE와 GRU head를 각각 30epoch까지 학습했다. 정상 validation 개선 기준을 통과한 가중치를 복원한다. 정상 global/patch/motion bank와 PCA도 추정했다. 모든 head의 학습 이력은 [feature_study_all16](results/feature_study_all16)에 있다. DINOv2는 동결했으며 GPT/VLM/GroundingDINO/object tracking은 이번 계열에 없다.

이 추가 head의 AUROC가 기존 CycleVAD를 넘지 못해 최종 균형 후보에는 포함하지 않았다. 모든 실패 설정도 공개한다. 16개 특징 설정과 39개 score 설정에서 중복 baseline 하나를 제외한 **54개 고유 설정**이며 54개 거대 모델을 학습한 것이 아니다.

## 최종 결과

아래의 초기/직전/최종 비교는 모두 우리 로컬 16장면 조건이다. 팀원 최신 4장면 CP와 직접 평균 비교하지 않는다.



| 설정 | AUROC | AP | FPR ↓ | Recall | Precision | F1 | Balanced acc. | Event coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 초기 고정 경보 | 67.28% | 41.37% | 10.12% | 17.24% | 30.53% | 18.58% | 53.56% | 43.24% |
| 직전 공통 경보 | 67.28% | 41.37% | 9.40% | 30.83% | 54.65% | 35.14% | 60.72% | 80.27% |
| 최종 max_mix_0.1 | 73.41% | 54.73% | 8.58% | 32.08% | 56.85% | 37.06% | 61.75% | 84.21% |

## 최종 후보의 16개 장면 수치

| 장면 | AUROC | AP | FPR ↓ | Recall | Precision | F1 | Balanced acc. | Event coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R01 | 83.57% | 62.62% | 36.94% | 70.81% | 49.72% | 58.42% | 66.94% | 100.00% |
| R02 | 97.30% | 94.29% | 1.77% | 66.54% | 95.08% | 78.29% | 82.39% | 100.00% |
| R03 | 63.96% | 53.46% | 6.36% | 11.72% | 57.39% | 19.47% | 52.68% | 64.71% |
| R04 | 67.17% | 71.58% | 3.30% | 14.46% | 84.87% | 24.71% | 55.58% | 53.85% |
| S01 | 72.97% | 66.06% | 9.34% | 29.06% | 66.80% | 40.50% | 59.86% | 60.00% |
| S02 | 64.34% | 30.30% | 10.79% | 27.02% | 38.42% | 31.73% | 58.12% | 100.00% |
| S03 | 82.85% | 63.00% | 1.08% | 17.11% | 84.56% | 28.47% | 58.02% | 80.00% |
| S04 | 59.88% | 38.99% | 23.49% | 27.44% | 39.09% | 32.24% | 51.98% | 100.00% |
| S05 | 97.11% | 93.02% | 2.40% | 94.00% | 61.04% | 74.02% | 95.80% | 100.00% |
| S06 | 56.94% | 33.88% | 5.80% | 18.38% | 43.38% | 25.82% | 56.29% | 77.78% |
| S07 | 71.18% | 42.78% | 2.64% | 21.14% | 61.65% | 31.48% | 59.25% | 90.91% |
| S08 | 74.04% | 37.86% | 2.40% | 1.49% | 13.85% | 2.69% | 49.55% | 50.00% |
| S09 | 79.29% | 54.12% | 7.81% | 47.32% | 61.82% | 53.60% | 69.75% | 87.50% |
| S10 | 80.17% | 57.16% | 8.18% | 30.91% | 53.86% | 39.28% | 61.37% | 100.00% |
| S11 | 57.90% | 29.46% | 12.12% | 12.63% | 27.27% | 17.27% | 50.26% | 90.91% |
| S12 | 65.83% | 47.08% | 2.93% | 23.16% | 70.84% | 34.91% | 60.11% | 91.67% |

S08 Recall 1.49%, F1 2.69%처럼 아직 사실상 대부분의 이상 프레임을 놓치는 장면이 있다. Event coverage 50%가 이 문제를 해결했다는 뜻은 아니다. 이벤트 한 구간에 경보가 한 번만 겹쳐도 event coverage에는 탐지로 계산된다.



## 성능 개선과 한계

- 최종 max_mix_0.1의 macro AUROC 73.41%, AP 54.73%, FPR 8.58%, Recall 32.08%, F1 37.06%는 같은 한 후보의 수치다.
- AUROC 최대 후보 mix_ema_0.5는 76.71%였지만 Recall 29.75%로 직전 30.83%보다 낮아 균형안에서 제외했다.
- R04와 S10의 AUROC는 하락했다. R01 Recall/F1 하락과 S08의 낮은 recall도 공개한다.
- 최종 결합 계수는 실제 테스트 비교 후 선택했다. 학습·보정·임계값에 test label을 안 썼다는 사실과 독립 최종 검증은 다르다.
- 정지 모듈은 phase별 16bin motion lower-tail이다. phase별 별도 PCA covariance나 LLM 공정 grammar가 아니다.
- 초기 motion 함수는 lag4의 앞 4값을 다섯번째 관측에서 복사한다. [재현 audit](scripts/audit_motion_prefix.py), [결과](results/motion_prefix_audit.json). 수정 뒤 보정·16장면 재평가가 필요하다.
- 기존 자동 테스트 63개는 이번 검토에서도 통과했다. [검사 기록](results/review_tests_20261012.txt). 이 suite는 위 초기 미래 참조를 검사하지 않는다.
- 정상 q99가 새로운 정상 영상에서 1% FPR을 보장하지 않는다. 원본 recording 독립성과 실제 cycle 위치 정확도는 미검증이다.
- 팀원 E16 FPS는 RTX PRO6000 96GB의 JPEG 입력이다. 우리 RTX4060의 현장 실시간 지표로 사용하지 않는다.

## 재현



Windows PowerShell 예시다. 실제 data root를 지정해야 하며 데이터와 사전학습 weights는 공개 저장소에 포함되지 않는다.

```powershell
git clone https://github.com/b-jongwon/cyclevad-adaptive-stall-ad.git
cd cyclevad-adaptive-stall-ad
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-local.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pytest -q
```

환경 버전은 `requirements-local.txt`와 팀원 E0 lock, 실제 encoder identity를 함께 확인한다. 기존 캐시가 다른 torch/preprocess/encoder identity면 섞지 말고 새 run 폴더에서 다시 추출한다.

```powershell
# 정상 CycleVAD fit / reference / threshold / testing
.\.venv\Scripts\python.exe -m cycle_vad.pipeline --data-root '<IPAD_dataset>' --config configs/local_all16_replay.json --output runs/base_all16 --stage all

# 정상 validation에서 공통 경보 선택 → 테스트 평가
.\.venv\Scripts\python.exe -m cycle_vad.universal_alarm --data-root '<IPAD_dataset>' --run-root runs/base_all16 --output runs/alarm_all16

# 16장면 AE/GRU·memory 및 16개 특징 설정
.\.venv\Scripts\python.exe -m cycle_vad.feature_study --data-root '<IPAD_dataset>' --run-root runs/base_all16 --alarm-root runs/alarm_all16 --output runs/features_all16

# 39개 score fusion / duration 설정 및 bundle
.\.venv\Scripts\python.exe -m cycle_vad.ranking_fusion --run-root runs/base_all16 --alarm-root runs/alarm_all16 --output runs/fusion_all16
```

이 명령은 코드가 노출한 CLI 인자에 맞춰 작성했으며, 현재 문서 작업 중 전체 학습을 다시 실행한 것은 아니다. 저장된 encoder/통계 버전 및 랜덤 샘플 조건이 완전히 일치하지 않으면 이전 결과와 소수점까지 같다는 보장은 없다.

## checkpoint로 추론

```python
import joblib
from cycle_vad.data import load_cache
from cycle_vad.ranking_fusion import predict_fusion

bundle = joblib.load('runs/fusion_all16/R01/balanced_pipeline.joblib')
data = load_cache('runs/base_all16/cache/R01/testing/01.npz')
prediction = predict_fusion(data, bundle)
print(prediction['score'], prediction['alarm'])
```

joblib/pickle은 신뢰하는 로컬 파일만 읽어야 한다. unknown source checkpoint는 임의 코드 실행 위험이 있다. 동영상 파일을 넣으면 이 함수가 자동으로 모든 전처리를 하는 구조가 아니라, 동일 encoder/stride로 생성한 feature cache가 입력이다.



## 결과·학습 증거

| 저장물 | 내용 | 출처 |
| --- | --- | --- |
| [results/E0–E17](results) | 기존 팀원 실험 | 팀원 원본, 결과값 변경 없음 |
| [adaptive_final](results/adaptive_final) | 초기 R01–R04 임계값·정지 탐색 9설정 | 로컬 확장 |
| [universal_all16](results/universal_all16) | 전체 16공통 경보·정상 validation 24후보 | 로컬 확장 |
| [feature_study_all16](results/feature_study_all16) | memory·AE·GRU 16설정/전체 학습 이력 | 로컬 확장 |
| [ranking_duration_all16](results/ranking_duration_all16) | score/duration 39설정·선택 한계 | 로컬 확장 |
| [전체 비교](docs/review/07_results.md) | 원본 JSON에서 생성한 지표 | 이번 정리 |
| [UPSTREAM_README](docs/UPSTREAM_README.md) | 원본 README 스냅샷 | upstream 075e21d |

공개 범위는 코드·설정·숫자·문서다. 데이터셋·특징 cache·joblib/pt 대형 checkpoint·API 키·자격 증명은 올리지 않는다. checkpoint는 사용자 로컬에 남아 있고 새 환경에서는 위 절차로 생성한다. 기존 upstream에 라이선스 파일이 확인되지 않아 임의로 재라이선스하지 않았으며 외부 재사용 조건은 원 저작자에게 확인해야 한다.

## 논문에 쓰기 전

초기 motion causality 수정 → normal 보정 재생성 → 동일 baseline/동일 split ablation → 후보·계수 사전 고정 → 독립 recording/외부 공정 확인 → 여러 seed·영상 단위 CI → 실제 정지/정상 대기 주석 및 오경보/지연 측정 순서가 필요하다. [연구 기여와 검증 계획](docs/review/08_paper_reproduction.md).

