# 논문에서 주장할 수 있는 것

## 가능한 연구 질문

“정상 공정에서 학습한 cycle-conditioned AD가 새로운 정상 영상에서 높은 점수를 만들고, 외형이 정상과 유사한 저움직임 이상을 놓칠 때, 정상 조건의 움직임 기준과 영상 강도 보정이 탐지 순위 및 경보 품질에 어떤 영향을 주는가?”

이 질문은 단순히 모듈을 여러 개 이어 붙였다는 것보다 실패 원인과 개선 가설을 명확히 한다. 최종 성능뿐 아니라 어떤 정보가 보완적이고 어떤 정보가 불필요했는지를 검증하는 논문 스토리를 만들 수 있다.

## 기여 후보와 현재 증거

| 기여 후보 | 현재 증거 | 아직 필요한 검증 |
| --- | --- | --- |
| score 수준 이동에 대한 경보 보정 | score 불변인 16장면에서 FPR/Recall/F1 변화 | 새 정상 촬영·초기 이상 오염·보정량 상한 |
| 정상 위치별 움직임 부족 보완 | lower-motion score 결합에서 평균 AUROC/AP 향상 | 실제 정지/정상 대기 주석, 초기 causal 수정 |
| 동일 규칙의 여러 공정 적용 | 16장면 동일 gain/결합 공식, 개별 정상 분포 | unseen scene 추가, 공정별 데이터량/적응 시간 |
| 최소 구조와 음의 ablation | AE·GRU·memory 확장보다 기존+stall 후보가 나음 | 여러 seed·신뢰구간·동일 학습 예산 |
| 운영 품질 개선 | frame FPR/recall/F1/event coverage | 시간당 경보·탐지 지연·현장 비용 함수 |

현 단계는 검증할 만한 연구 가설과 탐색 증거가 있는 것이다. 새 알고리즘의 novelty가 확정되었거나 국제 논문 대비 우월함을 입증한 상태는 아니다. 관련 연구에서 phase-aware motion/stall detection·adaptive calibration이 이미 어떻게 다뤄졌는지 문헌 비교가 추가로 필요하다.

## 논문 전에 필요한 통제 ablation

기본 feature·split·confidence 정책·normal calibration·test set을 고정하고 아래를 비교한다. 팀원 최신 CP와 초기 로컬 Full을 바로 비교하는 것은 ablation이 아니다.

1. 동일 baseline만 사용.
2. baseline + 영상 강도 적응형 임계값만 사용.
3. baseline + 정지 경보 OR만 사용.
4. baseline + 정지 연속 score만 결합, 임계값 보정 없음.
5. baseline + 정지 score + 강도 보정.
6. 5번 + 별도 연속 정지 OR.
7. 위치별 lower-tail vs global lower-tail.
8. lag1만 / lag4만 / min 결합, EMA 없음 / 0.5.
9. max vs 평균 결합, duration 사용/미사용.

후보별 학습·추론 비용도 같은 환경에서 기록한다. 모든 설정에서 frame AUROC/AP와 frame alarm 지표를 함께 보고, event coverage·지연·normal window FAR를 추가한다. 이벤트 탐지 한 번으로 프레임 성능을 대체하지 않는다.

## 우선 처리할 검증 순서

첫째, `_motion` 초기 미래 참조를 수정하고 prefix invariance test를 추가한다. 둘째, test 비교 후 고른 계수를 고정해 사전 명세에 기록한다. 셋째, 독립 recording/외부 공정에서 재평가한다. 넷째, 실제 cycle 경계와 정상 대기/이상 정지 구간을 소량이라도 주석해 모델의 해석을 검증한다.

여러 seed 42/43/44와 영상 단위 bootstrap CI를 사용하면 결과의 변동성을 더 잘 설명할 수 있다. 이 최종 로컬 후보는 현재 seed 42 결과이며 CI가 계산된 것처럼 쓰면 안 된다. 이미 확인한 테스트를 다시 나눈다고 과거 후보 선택의 영향을 완전히 지울 수 없다.

## 이번 CycleVAD와 과거 멘토 VLM 가이드의 차이

과거 가이드의 LLM/VLM은 객체 vocabulary·공정 grammar 생성, 상태/phase 추정, process consistency의 의미 판단 역할이었다. 현재 CycleVAD는 이를 수치 특징과 연속 θ 추적으로 대체한 별도 계열이다.

현재에는 객체 vocabulary, GroundingDINO bbox, object trajectory, VLM 언어 상태 추정, LLM semantic score가 없다. 따라서 이번 결과로 LLM의 효용이나 설명가능성을 입증할 수 없다. 정상 위치별 motion 분포와 연속 평균을 사용했다고 멘토 1–13단계를 모두 충실히 구현했다고도 할 수 없다.

## 설치와 기본 재현

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

## 코드와 결과 위치

`src/cycle_vad/`: 정상 model과 로컬 추가 모듈. `tests/`: 기존 63개 검사. `configs/local_all16_replay.json`: 전체 장면 baseline 재현 설정. `results/adaptive_final`: 초기 4장면. `results/universal_all16`: 공통 경보. `results/feature_study_all16`: 정상 neural head 이력. `results/ranking_duration_all16`: 결합 후보와 최종 선택 상태. `results/E0`–`E17`: 팀원 결과, 출처 분리.

로컬 모델·cache는 원래 실행 폴더에 보존했다. 새 공개 GitHub에는 코드·설정·숫자 결과·문서만 올린다. IPAD 이미지/ZIP·API 키·개인 자격 증명·대형 checkpoint를 공개하지 않는다.

## 출처와 논문 인용

기본 CycleVAD 및 기존 결과의 출처는 팀원 저장소다. IPAD 데이터, DINOv2, DTW/PCA/Fourier/GRU/autoencoder 등 선행 방법의 출처도 논문에서 인용해야 한다. 데이터 및 모델의 이용 조건은 각 배포처 조건을 별도로 확인한다. upstream에 확인된 라이선스 파일이 없으면 임의로 전체 코드를 MIT라고 선언하지 않는다.

논문에는 “팀원 기존 구현 기반 확장”, “정상 데이터만 학습·보정”, “테스트 비교 후 선택한 탐색 후보”, “미검증 온라인 초기화와 독립성”을 명시한다. 이 사실을 숨기지 않는 것이 결과를 신뢰받게 만드는 핵심이다.
