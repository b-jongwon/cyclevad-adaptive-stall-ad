# 후속 실험 명세: 추적 검증 → Confidence 분리 → 진행 이상 중심 기여

설계 고정 시점의 상태: **설계 완료, 실험 미실행, 주석 미작성**. 2026-10-09 KST.
현재 실행 결과는 [후속 실험 결과](FOLLOWUP_RESULTS.md)와 [실행 상태](../results/followup_status.json)를 따른다. 아래 계획 및 configs의 planned 상태는 실행 전 스냅샷이다.
E8 이후의 [개정 계획](PHASE_PROCESS_EXPERIMENTS.md)은 위치 조건 검증 E8B를 추가하고 합성 E9S를 E7 주석과 독립적으로 진행하도록 바꾼다.
이 문서는 E0–E5를 본 뒤 작성한 후속 실험 계획이다. 기존 IPAD 테스트 결과를 독립적인 확인 실험으로 재분류하지 않는다.

## 1. 고정 조건과 데이터 사용

- DINOv2-base checkpoint, 입력 크기, 전처리, 지역 특징은 E0–E5와 동일하게 고정한다. `configs/experiments/replay_pinned.json`과 기존 특징 캐시를 사용한다.
- 1차 model seed=42. 설정을 고정한 비교만 43/44로 반복하며, seed 평균은 데이터 분할 검증을 대신하지 않는다.
- 정상 영상 111개를 장면별로 5개 outer fold로 나눈다. 각 영상은 한 번만 `normal_evaluation`에 들어간다. 나머지는 FIT/validation/reference/threshold로 분리한다.
- 분할 단위는 현재 확인 가능한 **영상 파일**이다. 원본 녹화 그룹이 확인되면 동일 그룹으로 재구성하고 전체 계획 버전을 올린다. 파일 분리가 원본 녹화 독립성을 보장하지 않는다.
- 변형 영상과 원본은 항상 같은 outer fold에 둔다. 합성 이상 생성 규칙과 강도는 결과를 보기 전에 고정한다.
- 정상 reference로 component 스케일을 보정하고, 별도 정상 threshold의 q99로 각 최종 점수의 임계값을 정한다. 정상 평가 영상은 두 보정에 사용하지 않는다.
- 기존 테스트는 fold 0 모델의 **역사적 데이터 진단**으로만 평가한다. 라벨 불일치 R02 영상 12/13/14는 strict-v1에 따라 제외한다. 테스트 지표로 최종 후보·임계값을 선택하지 않는다.
- E8의 요소 효과는 같은 새 fold의 C-00과 비교한다. E0–E5와 분할이 다른 모델의 차이를 Confidence 효과라고 해석하지 않는다.
- 온라인 입력에는 현재·과거 관측과 실제 출력 프레임 번호만 제공한다. 테스트 영상 전체 길이, 편집에 사용한 원본 좌표, 주석 위치를 전달하지 않는다.

### Fold 0의 실제 분할 규모

| 장면 | FIT | Validation | Reference | Threshold | Normal evaluation |
|---|---:|---:|---:|---:|---:|
| R01 | 15 | 4 | 4 | 4 | 7 |
| R02 | 12 | 4 | 4 | 4 | 6 |
| R03 | 8 | 3 | 3 | 3 | 5 |
| R04 | 11 | 3 | 3 | 3 | 5 |

정확한 영상 ID는 `configs/experiments/followup/normal_folds.json`에 고정했다.

## 2. E6 — Cycle 가정·R01 보정 문제 진단과 주석 준비

### E6-A: 주석 pilot 40개 영상

각 장면에서 해시 순서로 FIT 2개, validation 2개, 정상 outer evaluation 4개, 유효 라벨을 가진 기존 이상 영상 2개를 선정한다. 총 40개이며 점수나 성공 사례로 고르지 않는다. 정확한 목록은 `annotation_queue.json`에 있고, 모든 주석값은 현재 null이다.

FIT 영상에서 장면별로 식별 가능한 반복 사건 3–6개를 정의한다. 사건 수를 억지로 맞추지 않는다. 각 영상에 다음을 기록한다.

- 완결 cycle의 시작·끝, 불완전 cycle, 실제 cycle 수.
- 공통 사건 ID와 발생 프레임, 정상적인 대기/속도 변화, 식별 불가능한 구간.
- 이상 구간의 유형: 외형 / 정지 / 역행 / 생략 / 순서 / 복합 / 판정 불가. 기존 binary 라벨을 이 유형의 정답으로 자동 변환하지 않는다.
- 분기 공정, 비주기적 구간, 여러 비동기 기계 등 단일 원형 좌표가 부적합한 사례.

모델 예측을 보지 않은 독립 검토로 사건 ID와 경계를 확인하고 불일치·보류를 남긴다. 모델의 DTW 정렬이나 프레임 번호/영상 길이를 실제 위치 정답으로 쓰지 않는다. 주석이 없는 동안 실제 추적 정확도를 수치로 주장하지 않는다.

### E6-B: R01 오경보 진단

각 component 및 최종 점수에 대해 reference / threshold / 정상 outer evaluation / 기존 테스트의 정상 프레임 분포(p50/p90/p99)를 비교한다. 영상별 FPR과 frame-weighted FPR을 함께 기록한다. 정상 검증에서 재구성 오차가 높다는 이유로 문제 영상을 삭제하지 않는다.

그림: 점수 분포 ECDF, 영상별 FPR, split별 normal score boxplot. 테스트를 보고 임계값을 올리는 실험은 하지 않는다.

## 3. E7 — 위치 추적과 신뢰도 검증

모든 arm은 같은 정상 FIT template, 특징, 최초 관측, FIT에서 정한 좌표 원점을 사용한다.

| ID | 방법 | 목적 |
|---|---|---|
| T0 | 첫 관측으로 초기화한 위치 + 정상 FIT 중앙 주기의 시계 | 단순 시간 경과 기준선 |
| T1 | 같은 causal descriptor의 template likelihood 원형 평균; 재귀 prior 없음 | Bayesian 누적 필터링이 필요한가 |
| T2 | 현재 causal Bayesian filter | 제안된 추적 방식 |

T1과 T2 모두 현재 구현의 과거 차분 특징을 사용한다. T1은 모든 시간 정보를 제거한 모델이 아니라 재귀적인 위치 분포 누적만 제거한 대조군이다. 원점과 사건별 목표 좌표는 FIT 주석의 사건을 학습 template에 대응시켜 모든 arm에 공통으로 고정한다. 시험 영상마다 최적 offset/DTW로 예측을 정답에 사후 정렬하지 않는다. 상태는 영상 시작에서 초기화하고 모든 arm의 초기 구간도 보고한다.

### 주지표와 정답의 범위

- 실제 사건 anchor에서 circular absolute error: `min(|pred−target|, 1−|pred−target|)`. 각 anchor의 목표 좌표는 FIT template에서 고정한 해당 사건 위치다. 사건을 임의의 등간격 k/K에 배치하지 않는다.
- anchor 사이에는 정확한 연속 위치를 추측하지 않고, 해당 **단계 구간에 예측이 들어가는 비율**을 보고한다.
- causal 예측이 해당 anchor를 전진 방향으로 처음 통과한 프레임과 실제 사건 프레임의 차이. 매칭은 사건 순서대로 일대일 수행하고 누락은 별도 보고한다.
- Confidence 순서로 남긴 25/50/75/100% anchor의 위치 오차(risk–coverage). Confidence를 확률이라고 부르거나 보정 전 ECE를 보고하지 않는다.
- 영상별/장면별 주석 coverage, 모호한 구간, cycle 수를 함께 제시한다. 보간 위치 오차는 보조 proxy로만 표시한다.

Pilot의 운영상 목표는 anchor MAE ≤0.05 cycle이고 T1 대비 MAE 20% 이상 감소하는 장면이 3/4 이상인 것이다. 이는 문헌 표준이 아니라 이번 연구의 진행 판단 기준이다. 넓은 신뢰구간은 성공이 아니라 미확정으로 판정한다. Pilot이 해석 가능하면 fold 0의 나머지 정상 evaluation 영상도 전부 주석해 결론을 확인한다. 단일 cycle 가정이 틀린 장면은 숨기지 않고 별도로 보고한다.

그림: anchor 예측 타임라인, 속도/시작점 조건별 오차, risk–coverage.

## 4. E8 — Confidence의 학습 효과와 점수 억제 효과 분리

코드에서 Confidence는 두 군데에 사용된다.

1. Fourier 평균 μ(θ) 학습의 가중치: `max(confidence, 0.1)`.
2. 추론의 조건부 이상 점수 곱셈: `confidence × conditional_score`.

기존 no_confidence 실험은 2번만 제거했다. 이번에는 아래 3×3을 모두 비교한다.

`r = |Σ likelihood_j exp(2πiθ_j)|`, `a = exp(−d_min/(4τ))`, 현재 Confidence는 `r*a`다. `r`도 정확도의 확률이 아니라 위치 후보 집중도다.

| 학습 가중치 ↓ / 추론 gate → | 현재 r*a | 위치 집중도 r | gate 없음 1 |
|---|---|---|---|
| 현재 max(r*a,0.1) | C-00 | C-01 | C-02 |
| 집중도 max(r,0.1) | C-10 | C-11 | C-12 |
| 균등 1 | C-20 | C-21 | C-22 |

- **점수 억제의 직접 비교:** C-00/C-01/C-02. 학습된 μ, PCA, θ가 동일하다.
- **학습 가중치의 직접 비교:** C-01/C-11/C-21. 추론 gate가 동일하다.
- 결과를 보기 전 지정한 후속 후보는 C-21, 단순 대조군은 C-22다. 기존 테스트 최고점 조합을 자동 선택하지 않는다.
- 같은 fold의 tracker, θ, pooled/local 분기와 rank 상한을 고정한다. C-00에서 정상 validation으로 선택한 Fourier 차수/ridge를 나머지 두 학습 arm에도 고정한다. μ와 그 잔차 PCA는 각 학습 arm에서 다시 학습한다.
- 각 학습 arm의 component calibration을 새로 계산하고 각 readout/gate의 최종 q99 임계값을 별도로 계산한다.

### 모든 cell에서 세 가지 readout

1. 조건부 점수 단독.
2. `max(appearance, gated conditional)` — **기여 ②의 주 평가 readout**.
3. `max(appearance, gated conditional, process)` — Full에서 효과가 가려지는지 확인.

원본 conditional 점수, gate 적용 후 점수, appearance 초과 비율, 정상·이상별 점수 분포, AUROC/AP, 정상 outer-evaluation FPR, Recall을 기록한다. 낮은 Confidence와 이상 발생의 상관만으로 원인을 확정하지 않고 동일 모델의 gate 개입 결과로 메커니즘을 판단한다.

Fold 0은 **12개 학습 모델(3×4장면)**로 36개 조합과 108개 readout 결과를 계산한다. 5-fold 전체는 최대 60개 통계 모델이며 backbone을 재학습하지 않는다. 같은 row의 추론 gate 3개는 추가 모델 학습 없이 계산한다.

실용적 진행 기준: C-21이 같은 fold의 C-00 대비 역사적 테스트 Macro AP를 1pp 이상 높이고, 정상 평가 FPR 증가는 각 장면에서 1pp 이하여야 한다. 단, 이는 개선 후보를 연구할 기준이지 독립 데이터에서 개선이 입증됐다는 뜻이 아니다. 상대적인 FPR 증가 제한을 만족해도 R01 절대 오경보 문제가 해결됐다고 하지 않는다.

그림: 3×3 AP heatmap(각 readout), gate 전후 점수, Confidence–위치 오차, branch activation.

## 5. E9 — 진행 이상 탐지의 직접적인 증거

### E9-A: 정상 프레임 기반 통제 실험

각 outer fold의 정상 evaluation 영상만 원본으로 사용한다. validation 영상의 편집본은 개발용으로만 쓰며 FIT/보정/평가에 섞지 않는다. 원본별 모든 편집본은 같은 분할에 둔다.

- 정지: 같은 관측을 L frame 반복 삽입.
- 역행: 길이 L 구간의 프레임 순서를 반전. 원본 프레임 집합은 유지.
- 생략: 길이 L 구간을 제거해 전진 점프를 생성.
- 순서 변경: 인접한 길이 L 구간 두 개를 교환. 원본 프레임 집합은 유지.
- L은 **정상 FIT 중앙 주기 T의 5/10/20%**, 삽입 시작 후보는 출력 타임라인의 0.25/0.50/0.75T. 정수화는 round-half-up, 최소 2프레임. 같은 규칙을 모든 arm에 적용한다.
- 관측 순서를 바꾼 뒤 tracker용 시간 차분과 posterior를 다시 계산한다. 정상 원본에서 계산한 tracker 상태를 편집본에 재사용하지 않는다.
- 출력 프레임 번호는 0부터 증가한다. 편집 전 프레임 번호/원본 위치 map은 평가기에만 제공한다.
- 충분한 prefix/편집 길이/평가 구간이 없는 조합은 제외 사유와 개수를 기록한다. 길이를 임의로 줄여 강도를 바꾸지 않는다.

정지·생략으로 영상 길이가 바뀌는 것 자체가 점수 max에 영향을 주지 않도록, 모든 원본/편집 쌍에서 **편집 시작부터 길이 0.4T인 동일 크기 구간**의 최대 점수를 window statistic으로 사용한다. 모델 상태는 영상 시작부터 causal하게 누적한다. 편집 시작 후 0.2T 이내 경보를 event detection으로 센다. 정지/생략의 평가 구간은 물리적 frame anomaly GT라고 부르지 않는다.

속도 stress는 전체 영상의 0.8/1.0/1.2배 재생 속도 조건을 따로 둔다. 허용 가능한 장비 속도 범위가 확인되기 전에는 '정상 산업 공정 FPR' 대신 **속도 stress 경보율**이라고 보고한다. 실제 정상 여부가 확인된 자연 속도 변화는 별도 normal 평가다.

### E9-B: 최소 대조군 6개

| ID | 점수 | 목적 |
|---|---|---|
| P0 | Appearance | 외형 기준선 |
| P1 | Appearance + alignment | 단순 template 이탈만으로 설명되는가 |
| P2 | Appearance + innovation | 전이 예측의 불일치 |
| P3 | Appearance + progress | 진행량 불일치 |
| P4 | Appearance + 세 process 점수 | 중심 후보 |
| P5 | Appearance + AR(1) 특징 전이 오차 | 일반적인 시간 정보보다 cycle 모델이 필요한가 |

AR(1)은 같은 FIT PCA 특징에서 원본 2프레임 간격의 ridge 특징 예측기를 학습한다. Ridge 후보는 0.01/0.1/1이고 정상 validation 오차로 고른다. AR(1)도 reference 보정과 별도 정상 threshold를 사용한다.

②가 개선되더라도 E9의 주 비교는 **P4 vs P0/P5**로 고정해 ③의 효과를 분리한다. 조건부 외형을 추가한 모델은 보조 비교다. 알려진 정지·역행·생략 유형은 실제 영상 주석에 기반해 따로 평가하며, 편집 실험 성능으로 실제 이상 유형별 성능을 대신하지 않는다.

주지표: 원본/편집 고정 window AUROC, 유형·강도별 event detection rate, 독립 정상 평가 frame FPR 및 동일 길이 정상 window 경보율. 부지표: 실제 positive 비율을 명시한 AP, 탐지된 이벤트의 지연과 미탐지 수. 동일 원본에서 파생한 많은 편집본을 독립 표본으로 세지 않는다.

실용적 진행 기준: P4가 P0와 P5 모두보다 event detection을 5pp 이상 높이고, 장면별 정상 outer-evaluation FPR이 5%를 넘지 않아야 한다. q99의 의도는 약 1%이며 5%는 연구 진행을 위한 느슨한 상한이다. 평균으로 나쁜 장면을 숨기지 않는다.

### E9-C: stride의 시간 범위 일치

stride 2/1에서 descriptor lag를 원본 4프레임, progress lag를 원본 [2,8,32]프레임, AR(1) 예측 간격을 원본 2프레임으로 맞춘다. 후보 C-21과 P4/P0/P5 설정을 test 결과에 맞춰 재선택하지 않는다. Backbone cache는 해당 stride에 맞게 검증한다. 표본 밀도에 따른 fitting/calibration 차이는 남으므로 완전히 동일한 모델의 순수 추론 샘플링 효과라고 주장하지 않는다.

그림: 유형×강도 heatmap, P4/P5 paired improvement, 속도 stress 경보율, source-frame-matched stride 비교.

## 6. 불확실성, 중단·전환 기준과 연구 기여

- 장면별 stratified paired bootstrap 2,000회, seed=20261009, 95% percentile interval. 원본 파일 또는 확인된 녹화 그룹 단위로 재표집하고 파생 편집본을 함께 묶는다. 프레임 독립 bootstrap은 사용하지 않는다.
- 작은 주석 pilot에서 넓은 구간이 나오면 미확정으로 남긴다. 파일 단위 결과를 독립 recording 통계로 과장하지 않는다.
- ①이 실패하면 cycle 가정과 추적기 개선이 먼저다. ③을 안정적인 cycle-aware 검출로 주장하지 않는다.
- ①이 유효하고 ②가 효과 없으면 ②를 보조/음성 결과로 남기고 **Appearance + Process**를 중심 방법으로 단순화한다.
- ②가 회복되면 기여 ③을 돕는 불확실성 처리로 제시한다. 단순히 새 조합이 최고점이라고 세 번째 독립 novelty를 추가하지 않는다.
- ③이 일반 전이 모델 P5를 이기지 못하면 기여를 일반적인 시간 정보 결합 수준으로 좁힌다.
- E10은 새롭고 독립적인 녹화 데이터가 확보된 뒤, 고정된 방법을 확인하는 단계다. 현재 데이터로 E10 완료를 선언하지 않는다.

## 7. 실행 순서와 산출물

1. **E6:** 정상 5-fold manifest, 40개 영상 주석 packet, R01 분포 진단. Manifest와 선정 목록은 이번에 작성했지만 진단/주석은 아직 수행하지 않았다.
2. **E7:** 주석 검토 후 T0/T1/T2 평가. 실제 주석 없이 추적 정확도를 대체 수치로 채우지 않는다.
3. **E8:** 9-cell Confidence 실험. 점수 재결합 진단은 주석 준비와 독립적으로 구현 가능하지만 최종 해석은 E7과 함께 한다.
4. **E9:** P0–P5 시간적 이상 실험, 자연 속도 변화 및 source-frame-matched stride 평가.
5. **E10:** 별도 녹화 확보 후 고정 후보의 확인 평가.

각 단계가 완료되면 `results/E6` 등의 수치 JSON/CSV, 사용한 split/config/코드 버전, 그림, 제외 목록을 저장하고 README를 갱신해 GitHub에 업로드한다. 계획 상태를 완료로 표시하거나 기존 E0–E5 결과를 새 실험 결과로 복사하지 않는다.

설정: [protocol.json](../configs/experiments/followup/protocol.json), [normal_folds.json](../configs/experiments/followup/normal_folds.json), [annotation_queue.json](../configs/experiments/followup/annotation_queue.json).
재생성: `python3 scripts/prepare_followup.py`.
