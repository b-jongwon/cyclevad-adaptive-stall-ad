# CycleVAD 논문 통합 실험 실행 규칙

2026-10-09에 고정한 첫 실행 묶음은 E11 통합 ablation, E12 평균·subspace 분리 비교와 head 표본 효율, E14A 기존 편집의 경계·내부 분석이다. [기계 판독 설정](../configs/experiments/followup/paper_stage1_v1.json)에 비교군·분할·보정·집계 규칙을 기록했다. [완료 결과](PAPER_STAGE1_RESULTS.md)에서 실제 실행 상태를 관리한다. 설정의 planned 상태는 실행 전 스냅샷이다.

- 기존 정상 5-fold와 seed 42/43/44, 고정 RTX PRO 6000 DINOv2 dense CUDA 특징을 사용한다. 새 통계 모델은 CPU 4 threads/job에서 학습한다.
- 기존 E9S의 fold/seed별 FIT-only PhysicalTracker 및 외형 모델을 공통으로 사용한다. E11의 조건부 평균은 새 tracker의 정상 validation에서 seed42에 Fourier 차수 2/4/8, ridge .01/.1/1 중 선택하고 다른 seed에 같은 선택을 적용한다. 모든 reference 보정은 새 계산에서 다시 구성한다.
- E11 A, A+C, A+P, A+C+P, C, P를 비교한다. C는 균등 평균 학습·confidence gate 없음, P는 alignment/innovation/progress의 고정 결합이다. 모델별 정상 window-q99와 고정 보조 quantile 곡선을 보고한다. 동일 실제 테스트 FAR을 보장하지 않는다.
- E11 주 비교는 Full−(A+P), Full−(A+C)이다. Historical fold0의 프레임 AP/AUROC, source별 이벤트와 합성 OOF 이벤트를 구분한다. 합성 3,910개/seed의 입력은 dense source 편집 후 output stride2이며 모든 시간 상태를 다시 계산한다.
- E12는 전역/연속/4·8-bin 평균과 공유/4·8-bin PCA를 분리하고 연속 평균의 norm-only 대조군을 포함한다. 공통 총 rank를 맞추되 실제 저장 bytes가 같다고 가정하지 않는다. 모든 PCA에서 variance 조기 절단을 끈다. 각 bin의 최소 표본·원본 수가 부족하면 사전 규칙대로 unavailable로 기록한다.
- E12 25/50/100% FIT 부분집합은 영상 해시 순서로 중첩한다. 추적기와 A, h/ridge는 full FIT/normal validation에서 고정되므로 head 표본 효율이지 전체 시스템 few-shot 평가가 아니다. Historical/normal 평가는 고정하며 process를 합치지 않는다.
- E14A는 원본과 같은 구간 길이의 정상 경보를 대조한다. 초기32프레임과 유효 내부는 표본·창 길이가 다르다. 경계에서 바뀐 filter 상태는 내부에도 남으므로 경계 독립성의 증거로 단정하지 않는다. Skip 내부는 평가하지 않는다. 새로운 대응 편집과 회복 실험은 별도 후속이다.
- 기존 historical test는 개발에 이미 사용됐고 정상 편집도 기존 원본에서 파생된다. 전체 결과는 탐색적이다. 2,000회 paired source bootstrap은 고정 모델 조건부 구간이며 seed/편집을 독립 영상으로 세지 않는다. 원 촬영 세션의 독립성은 아직 미확인이다.
- R02 testing12/13/14의 라벨 길이 불일치는 기존 strict-v1 전체 제외 정책을 유지한다. Historical 이상을 전부 외형 이상으로 간주하지 않는다.

재현 명령:

```bash
export PYTHONPATH=src
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python scripts/run_paper_stage1.py --stage E11 --resume
python scripts/run_paper_stage1.py --stage E12 --resume
python scripts/analyze_paper_boundary.py
python scripts/summarize_paper_stage1.py --stage E11 --seed 42 --bootstrap
python scripts/summarize_paper_stage1.py --stage E11 --seed 43
python scripts/summarize_paper_stage1.py --stage E11 --seed 44
python scripts/summarize_paper_stage1.py --stage E12
python scripts/check_paper_stage1.py --stage E11
python scripts/check_paper_stage1.py --stage E12
python scripts/report_paper_stage1.py --stages E11 E12 E14
```

E11 수치 재현 검사에서 float32 batch 길이에 따른 반올림이 경험적 reference 순위에 증폭되는 문제가 발견되어 C 점수 계산을 float64로 변경했다. 모델·분할·선택 기준은 바꾸지 않았고 전체 E11을 재실행했다. 편집 이전 score prefix와 tracker prefix의 불변성 및 저장 점수 재평가를 완료 조건으로 둔다.
