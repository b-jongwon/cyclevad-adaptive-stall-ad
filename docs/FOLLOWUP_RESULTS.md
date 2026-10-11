## 후속 실험 E6–E10

[사전 고정 실험 명세](FOLLOWUP_EXPERIMENTS.md) · [후속 검사 로그](../results/followup_tests.txt)

### E6 — 정상 분할·보정 이동 진단 완료

정상 111개 영상의 5개 분할(장면×fold 20개)에서 fit/validation/reference/threshold/evaluation 중복이 없고 각 영상이 정확히 한 번 평가됨을 확인했습니다. 파일 간 원본 recording 독립성은 아직 미확인입니다. 40개 영상의 모델 예측을 숨긴 로컬 프레임 뷰어와 접촉시트를 준비했습니다. **실제 cycle 주석은 0개이며 위치 정확도는 미측정**입니다.

| 장면 | Reference FPR % | Threshold FPR % | Normal eval FPR % | Historical normal FPR % | Eval 영상 평균 FPR % |
| --- | --- | --- | --- | --- | --- |
| R01 | 0.00 | 0.89 | 0.00 | 47.92 | 0.00 |
| R02 | 1.51 | 0.76 | 0.97 | 1.18 | 0.96 |
| R03 | 2.07 | 0.94 | 7.12 | 4.28 | 7.29 |
| R04 | 0.00 | 0.90 | 1.54 | 4.08 | 1.55 |

표는 fold 0의 C-00 Full, 각 분할의 정상 프레임 기준입니다. Reference/threshold는 보정에 사용했으므로 일반화 성능이 아닙니다. Historical은 기존 테스트의 정상 라벨 구간입니다. 정상 q99라도 새로운 정상 영상에서 1% FPR을 보장하지 않습니다. 테스트 정상 구간에는 앞선 이상으로 인한 추적 상태 영향도 있을 수 있어, 이 차이를 촬영 환경 변화만의 원인으로 확정하지 않습니다.

![정상 점수 분포](figures/E6_normal_ecdf.svg)

![영상별 오경보](figures/E6_video_fpr.svg)

R01 historical 정상 오경보 1165프레임 중 **98.45%**에서 pooled 외형 분기가 최댓값이었습니다. Confidence로 가중한 조건부 분기가 최댓값인 오경보는 0개였습니다. 이는 점수 귀속 진단이며 촬영 환경 등 원인을 확정하는 인과 분석은 아닙니다. [분기별 진단](../results/E6/R01_channel_diagnosis.json)

[E6 수치](../results/E6/calibration_summary.json) · [분할 검사](../results/E6/fold_audit.json) · [주석 자료 상태](../results/E6/annotation_packet.json)

### E8 — Confidence 3×3 실험 seed 42 완료

20개 장면×fold, 통계 모델 60개를 학습하고 각 모델의 추론 게이트 3종과 readout 3종을 평가했습니다. 각 fold의 추적기·θ·pooled/local 분기와 C-00에서 선택한 Fourier 차수·ridge를 고정했습니다. C-00 재학습 및 원래 점수와의 동등성 검사를 모두 통과했습니다.

C-행열: 학습 행 0=legacy, 1=위치 신뢰도 r, 2=균등; 추론 열 0=legacy, 1=r, 2=gate 없음. 아래 주지표는 max(appearance, conditional)입니다. 모든 조합과 readout은 각각 정상 q99 임계값을 사용합니다.

| 조합 | Historical macro AUROC % | Historical macro AP % | R01 정상 OOF FPR % | R02 정상 OOF FPR % | R03 정상 OOF FPR % | R04 정상 OOF FPR % |
| --- | --- | --- | --- | --- | --- | --- |
| C-00 | 76.26 | 68.01 | 5.49 | 1.14 | 4.84 | 1.77 |
| C-01 | 76.23 | 68.05 | 5.49 | 1.19 | 3.81 | 1.77 |
| C-02 | 77.51 | 70.50 | 5.48 | 1.19 | 3.97 | 2.12 |
| C-10 | 76.25 | 68.01 | 5.49 | 1.14 | 4.84 | 1.77 |
| C-11 | 76.22 | 68.05 | 5.49 | 1.21 | 3.81 | 1.77 |
| C-12 | 77.59 | 70.66 | 5.65 | 1.18 | 3.99 | 2.24 |
| C-20 | 76.25 | 68.01 | 5.49 | 1.14 | 4.84 | 1.77 |
| C-21 | 76.23 | 68.05 | 5.49 | 1.21 | 4.11 | 1.77 |
| C-22 | 77.82 | 71.10 | 5.40 | 1.18 | 4.08 | 2.22 |

![Confidence factorial](figures/E8_confidence_factorial.svg)

사전에 고정한 C-21의 C-00 대비 historical macro AP 변화는 **+0.039 pp**입니다. AP +1 pp 및 장면별 정상 FPR 증가 ≤1 pp라는 점추정 기준은 **미충족**입니다. 기존 테스트는 탐색적 진단이며 확증 자료가 아닙니다.

[전체 요약](../results/E8/summary.json) · 각 장면/fold의 metrics.json, per_video.json, mechanism.json에는 27개 readout과 gate 활성 비율·점수 분포가 포함됩니다. seed 43/44 재검증 결과는 아래에 정리했습니다.

![조건부 분기 활성 비율](figures/E8_gate_activation.svg)

R01은 일치도 항을 제거해도 r 자체가 낮고 조건부 점수가 외형 점수를 넘는 비율이 0%였습니다(C-00/C-01/C-21, historical 이상 프레임). 낮은 r이 실제 위치 오차인지 반복 외형의 모호성인지 구분하려면 E7 주석이 필요합니다.

| Readout | 조합 | Historical macro AUROC % | Historical macro AP % |
| --- | --- | --- | --- |
| conditional_only | C-00 | 69.12 | 57.43 |
| conditional_only | C-21 | 73.31 | 64.31 |
| conditional_only | C-22 | 79.40 | 72.43 |
| appearance_conditional | C-00 | 76.26 | 68.01 |
| appearance_conditional | C-21 | 76.23 | 68.05 |
| appearance_conditional | C-22 | 77.82 | 71.10 |
| full | C-00 | 78.46 | 70.55 |
| full | C-21 | 78.43 | 70.55 |
| full | C-22 | 79.60 | 73.16 |

사전 대조군 C-22(균등 학습·gate 없음)의 주 readout AP 변화는 **+3.090 pp**, 파일 bootstrap 95% 구간은 [+1.784, +4.916] pp입니다. C-21을 사후 교체하지 않았으며, gate 제거를 별도 가설로 검토할 근거입니다. 두 구간은 다중 비교 보정 없는 탐색적 구간입니다.

C-21−C-00 historical macro AP의 파일 단위 paired bootstrap 95% 구간: **[-0.161, +0.300] pp** (2,000회). 영상 원본 그룹 독립성은 미검증입니다. [불확실성 수치](../results/E8/paired_bootstrap.json)

### E8 — seed 43·44 고정 설정 재검증 완료

총 3 seeds × 4 scenes × 5 folds에서 통계 모델 180개, 조합별 readout 1,620개를 평가했습니다. Seed 42의 각 fold에서 선택한 Fourier 차수·ridge, 데이터 분할과 특징 투영을 고정했습니다. 표는 주 readout의 historical macro AP입니다.

| Seed | C-00 AP % | C-21 AP % | C-22 AP % | C-21 ΔAP pp | C-22 ΔAP pp |
| --- | --- | --- | --- | --- | --- |
| 42 | 68.01 | 68.05 | 71.10 | +0.039 | +3.090 |
| 43 | 68.55 | 68.63 | 71.77 | +0.083 | +3.225 |
| 44 | 68.62 | 68.58 | 71.45 | -0.044 | +2.829 |

같은 자료에서의 seed 민감도이며 독립 데이터 재현을 의미하지 않습니다. [재검증 수치](../results/E8/robustness.json)

재실행: `PYTHONPATH=src python scripts/run_followup.py --data-root /path/to/IPAD_dataset` 후 `--seed 43`, `--seed 44`로 반복합니다. `scripts/prepare_annotation_packet.py`는 로컬 원본 경로에서 주석 뷰어를 생성합니다. `scripts/bootstrap_followup.py`와 `--candidate C-22`로 paired CI를 계산하고, `scripts/check_followup.py`로 저장된 점수와 수치를 검증합니다. [정합성 검사](../results/followup_validation.txt)

E7: T0/T1/T2 인과적 예측을 로컬에 저장했으며 [독립적인 실제 cycle/anchor 주석](ANNOTATION_GUIDE.md)이 필요합니다. E9S: 합성 편집 검증은 E7 주석과 독립적으로 진행하며 별도 결과 문서에서 실행 상태를 관리합니다. 실제 유형별 E9R은 주석이 필요합니다. E10: 신규 독립 촬영 자료가 필요합니다. 미실행 항목을 완료로 표시하지 않습니다.

### 다음 실험 계획

[위치별 정상 기준과 진행 이상 검증 계획](PHASE_PROCESS_EXPERIMENTS.md)을 고정했습니다. E8B는 위치 조건 5개 비교군, E9S는 진행 점수 10개 비교군을 사용합니다. 편집 후보 3,996개 중 3,910개가 길이 검사를 통과했습니다. 실행 결과는 [E8B / E9S 결과](PHASE_PROCESS_RESULTS.md)를 참고하세요.

