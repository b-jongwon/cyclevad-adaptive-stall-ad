# Continuous Cycle VAD — Colab 실행 안내

이 구현은 기존 저장소를 바탕으로 추가한 **VLM을 사용하지 않는 독립 실행 경로**다. `cycle_vad`는 Qwen, text encoder, GroundingDINO, semantic phase 생성, phase별 PCA bank를 불러오지 않는다. 기존 `ipad_vad`와 과거 실험은 비교·재현용으로 남아 있지만 새 Colab 경로의 의존성이 아니다.

## 바로 실행

1. `notebooks/CycleVAD_Colab.ipynb`를 Google Colab에 업로드한다.
2. 런타임 → 런타임 유형 변경 → GPU를 선택한다. T4를 기본 설계 대상으로 삼았지만 실제 Colab GPU에서의 실행은 아직 미측정이다.
3. 설치 셀과 Drive 연결 셀을 실행한다. **노트북에 구현 소스가 내장되어 있으므로 GitHub에 변경사항을 push할 필요가 없다.**
4. `DATA_ROOT`를 이미 압축 해제한 IPAD 폴더로 설정한다. 기본값은 `/content/drive/MyDrive/IPAD_dataset`이다.
5. 설정, 선택적 로컬 복사, 검증, 학습·평가, 결과 시각화 셀을 순서대로 실행한다.

필요한 데이터 구조:

```text
IPAD_dataset/
  R01/
    training/frames/01/0.jpg, 1.jpg, ...
    testing/frames/01/0.jpg, 1.jpg, ...
    test_label/001.npy
  R02/ ...
```

기본 장면은 R01~R04. `SCENES`에 S01~S12를 넣으면 합성 장면도 동일하게 처리한다. 프레임 이름은 정수이며 0부터 빠짐없이 이어져야 한다. 라벨 길이가 다르면 해당 테스트 영상 전체의 평가 라벨을 unknown으로 처리한다. 자르거나 늘려서 맞추지 않는다. 추론과 점수 저장은 unknown 여부와 무관하게 수행한다.

## 구조

```text
영상 → frozen visual backbone → 전체 프레임 + 여러 층의 지역 특징
                                  ├→ 정상 cycle template + causal Bayesian tracker
                                  │       → 연속 angle, confidence, 진행 불일치
                                  ├→ cycle 조건부 Fourier 평균 μ(angle)
                                  │       → 잔차의 공유 PCA subspace 하나
                                  └→ pooled PCA + cycle 전체 공유 local memory
                                                ↓
                                정상 reference로 점수 보정 → 결합
                                                ↓
                                별도 정상 holdout으로 경보 임계값 결정
```

- 기본 백본: `facebook/dinov2-base`, 336×336, 마지막/세 번째 마지막 hidden layer, 6×6 지역 격자, 64차원 고정 random projection. 실제로 내려받은 checkpoint revision과 환경 버전을 기록한다.
- 넓은 산업 영상의 가장자리를 자르지 않도록 전체 프레임을 letterbox한다. 의미적 객체 이름 대신 고정된 지역 위치를 활용한다. 카메라 이동에 강하다는 보장은 없다.
- FP16은 CUDA 추출에만 사용하고 CPU 통계 모델은 FP32/FP64를 사용한다. batch 8에서 메모리가 부족하면 4→2→1로 자동 재시도한다. batch 1에서도 실패하면 입력 크기/백본을 줄이라는 오류를 낸다.
- 학습 시 정상 영상의 template를 DTW로 정렬한다. 테스트에서는 미래를 보지 않는 Bayesian filtering만 사용한다. 추론에 테스트 영상 전체 길이 또는 정답 cycle 위치를 사용하지 않는다.
- `cycle_grid=128`은 원 위 확률 분포의 수치적 근사다. 128개 독립 phase 모델이나 bank를 만들지 않는다. 출력 angle은 원형 평균을 통해 연속값으로 얻는다.
- 외형 모델은 `z = μ(angle) + U a + ε`다. Fourier 평균과 cycle 전체가 공유하는 잔차 subspace를 학습한다. subspace 밖 오차와 내부 좌표의 Mahalanobis 크기를 모두 사용한다.
- 보조 pooled subspace와 local memory는 cycle 불확실성 때문에 외형 이상이 사라지지 않도록 한다. 이들도 cycle별 bank로 나뉘지 않는다. 지역 점수는 거친 격자 근거이며 객체/픽셀 GT localization 정확도를 의미하지 않는다.
- 정지·역행·작업 생략 신호가 tracker의 smoothing에 가려지지 않도록 관측 전 예측 불일치, 여러 시간 간격의 진행 오차, template 일치 오차를 따로 계산한다.
- 임계값이 percentile 1에 포화되는 문제를 피하려고 경험적 log-tail 점수에 reference 최대값 초과분을 연장한다. 결과는 보정된 점수이며 실제 이상 확률은 아니다.

## 실제 cycle 경계가 있는 경우

권장 입력은 정상 영상에서 사람이 확인하거나 장비 로그로 얻은 cycle 시작 경계다. 프레임 단위 JSON으로 지정한다:

```json
{
  "R01/training/01": [0, 180, 363, 540],
  "R01/training/02": [22, 204, 385]
}
```

각 인접한 두 값은 완결 cycle의 `[시작, 다음 시작)` 구간이다. 마지막 값은 exclusive end이며 프레임 수와 같아도 된다. 바깥의 불완전 구간은 cycle template 학습에서 제외된다. sampled 관측이 8개 미만인 cycle은 오류를 낸다. 서로 다른 영상의 cycle 시작점은 같은 공정 사건에 맞춰야 한다.

노트북의 `BOUNDARIES_JSON` 또는 CLI `--boundaries`로 파일을 지정한다. 실제로 선택된 FIT 영상의 경계만 template 학습에 사용된다. Test cycle 경계를 추론에 전달하지 않는다.

**경계가 없을 때:** 정상 영상 한 편을 한 cycle 후보로 놓고 영상 간 DTW 정렬을 학습하는 `weak_recording_alignment`로 실행한다. 영상이 정확히 한 cycle이라는 사실을 검증한 것으로 취급하지 않는다. 정상 녹화가 여러 cycle 또는 임의의 부분 구간이면 이 대체 방식이 틀릴 수 있다. `fit_report.json`에 출처를 모두 남기며 cycle accuracy를 산출했다고 주장하지 않는다. 이 경우 실제 경계 확인이 성능 개선의 우선 과제다. 단순히 테스트 프레임 번호를 영상 길이로 나누어 cycle을 부여하지 않는다.

서로 비동기적인 여러 기계, 갈래가 있는 작업 순서, 반복되지 않는 공정은 하나의 원형 좌표로 충분하지 않을 수 있다.

## 정상 데이터 분할과 모델 선택

장면별로 최소 8개의 정상 영상/원본 녹화 그룹이 필요하다. 고정 seed의 해시 순서로 약 55/15/15/15%를 분할한다(작은 표본에서 반올림).

| 분할 | 용도 |
|---|---|
| fit | cycle template, Fourier 회귀, shared/pooled subspace, local memory |
| validation | Fourier 차수·ridge를 정상 재구성 오차로 선택 |
| reference | 외형·진행 component별 점수 스케일 보정 |
| threshold | 최종 결합 점수의 q99 임계값 결정 |
| testing | 동결된 모델을 평가하며 학습·선택에 쓰지 않음 |

원본 녹화 그룹을 알고 있으면 `GROUPS_JSON`/`--groups`로 `{ "R01/training/01": "recording_A", ... }`를 제공한다. 같은 그룹이 서로 다른 분할에 들어가지 않도록 한다. 파일별 분할만으로 원본 그룹 독립성이 입증되는 것은 아니다. 사용자 제공 그룹의 진실성을 자동 검증했다고 표시하지 않는다.

`μ`의 후보 선택은 정상 분포 적합을 확인하는 절차이며, 실제 이상 탐지 성능 최적화를 보장하지 않는다. 기본 백본도 새 구조에서 최적임을 측정한 결과가 아니다. 저장소의 이전 실험에서는 단순 모델 확대가 성능 향상으로 이어지지 않았으므로 large 모델을 무조건 기본값으로 두지 않았다.

`appearance`, `cycle_conditioned`, `combined` 세 출력을 **각각의 정상 임계값**과 함께 평가한다. 이는 구성 요소의 진단이며, 테스트 결과를 보고 자동으로 가장 좋은 모델을 선택하지 않는다. 순수한 구조 대조를 위해 discrete-phase baseline 재실험까지 완료한 것은 아니다.

## Drive, 중단 후 재개, 실행 비용

- 폴더를 Drive에서 직접 읽을 수 있다. 다수의 작은 JPG를 읽는 병목을 줄이려면 노트북의 `STAGE_TO_LOCAL=True`를 사용한다. 처음 복사는 시간이 걸리며 Colab 로컬 디스크 용량을 확인해야 한다.
- 모델·캐시·점수·보고서는 `OUTPUT_DIR`(Drive)에 저장된다. 런타임이 종료되어도 완료한 영상의 캐시를 재사용한다. 미완료 영상은 처음부터 다시 추출한다.
- 캐시 키는 백본/전처리/stride/실제 checkpoint revision/프레임 파일명·크기·수정시간을 포함한다. 이미지 내용 전체 해시를 검사하는 것은 아니다.
- 캐시를 쓰던 백본 설정을 바꾸면 재추출한다. 서로 다른 특징 설정을 섞어 학습하거나 평가하면 오류를 낸다.
- 학습은 장면별로 처리하고 영상 길이를 균형 있게 샘플링해 메모리를 제한한다. `fit_samples`, `subspace_rank`, `local_memory`로 CPU 메모리·시간을 조절할 수 있다.
- `model.joblib`은 Python 객체 직렬화이므로 본인이 생성한 신뢰할 수 있는 checkpoint만 읽는다.

실제 Colab 실행 시간, T4 peak VRAM, pretrained checkpoint 다운로드를 포함한 end-to-end 비용은 아직 측정하지 않았다. Colab의 GPU/세션 자원은 계정·시점에 따라 달라진다.

## CLI

```bash
pip install -r requirements-cycle-colab.txt
pip install --no-deps -e .
python -m cycle_vad.pipeline \
  --data-root /content/drive/MyDrive/IPAD_dataset \
  --output /content/drive/MyDrive/IVAD_cycle_runs/run01 \
  --config configs/cycle_colab.json --scenes R01 R02 R03 R04
```

`--stage extract`, `--stage fit`, `--stage evaluate`로 분리 실행할 수도 있다. `fit`과 `evaluate`는 이미 추출된 캐시를 사용한다. 입력 이미지를 수정했다면 먼저 `extract`를 다시 실행해 캐시를 검증·갱신한다.

## 결과 파일

- `environment.json`, `encoder.json`, `config.json`, `manifest.json`: 실행 환경·입력·추출 설정.
- `R01/model.joblib`, `fit_report.json`, `protocol.json`: 모델, 정상 분할, 모델 선택, 사용한 캐시, 소스 코드 fingerprint.
- `R01/predictions/testing_01.npz`: sampled 원본 frame index, 점수, cycle angle/confidence, 지역 거리.
- `R01/metrics.json`, `per_sequence.csv`: AUROC/AP, 정상 FPR, 이상 recall, event coverage, 탐지된 event의 첫 경보 지연.
- `macro_metrics.json`: 실행한 장면별 지표의 단순 평균. 전체 공식 benchmark와 동일한 평가를 주장하지 않는다.

점수는 이전 sampled 값을 유지해 원본 프레임으로 확장한다. 미래 점수로 선형 보간하지 않는다. stride 때문에 짧은 이상을 놓칠 수 있으므로 최종 성능 평가에서 `stride=1`도 별도 검증할 수 있다. FPS를 가정하지 않고 지연은 원본 프레임 수로 표시한다. cycle 정확도와 pixel localization 정확도는 해당 주석 없이는 미측정이다.

## 구현 검증과 현재 한계

실제 데이터 성능 실험과 코드 검증을 구분한다. 로컬 테스트는 정상/이상 합성 특징, 작은 실제 DINOv2/CLIP 아키텍처, 임시 IPAD 형태 폴더를 사용한다. 사전학습 가중치에 대한 실제 IPAD/Colab 벤치마크는 아직 실행하지 않았다. 검증 결과는 `CYCLE_IMPLEMENTATION_REPORT.md`에 기록한다.

참고: [IPAD 논문](https://arxiv.org/abs/2404.15033), [DINOv2 공식 Transformers 문서](https://huggingface.co/docs/transformers/model_doc/dinov2), [CLIP 공식 문서](https://huggingface.co/docs/transformers/model_doc/clip), [Colab FAQ](https://research.google.com/colaboratory/faq.html).
