# 로컬 개선 3: 정지 증거를 연속 점수에도 반영

## 경보 OR만으로는 무엇이 부족했나

앞 단계는 기존 score를 유지하고 정지 경보만 OR했다. 이벤트를 더 찾더라도 기존 score로 평가한 AUROC/AP는 변하지 않는다. 정상 외형처럼 보이는 정지 구간이 높은 순위에 오르도록 **정지 점수도 최종 연속 score에 들어가야 한다**는 가설을 세웠다.

정지 점수는 자동 고장 설명이 아니다. 현재 관측의 특징 변화가 정상 해당 위치의 변화보다 드문 수준으로 작은지를 수치화한다. 전체 화면 global 특징이므로 작은 부품의 정지만으로 충분히 변하지 않거나 조명이 흔들리는 경우에는 놓칠 수 있다.

## 학습/기준 추정 순서

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

## 함께 시험한 결합들

**weighted mix:** `(1−w)b+w×u`. **mix EMA:** 이 점수를 alpha 0.4로 시간 평활화한다. **log mix:** log1p(b), log1p(u)를 평균 결합한다. **max mix:** `max(b,(1+w)u)`이다.

w는 0.05/0.1/0.2/0.3/0.5/0.75를 비교했다. 여기에 baseline, base EMA, stall만의 ranking, 지속 시간 add/max/gated 후보를 더해 39개 설정이다.

duration은 u가 0.8을 초과한 연속 길이를 센다. 정상 reference의 해당 길이 q99.5(최소 3관측)로 나눠 평소보다 긴 저움직임 구간을 읽는다. 정상 대기보다 오래 정지하면 점수를 늘리겠다는 가설이다. duration 후보 중 일부는 좋은 AUROC를 보였지만 최종 균형 후보에는 들어가지 않는다.

## 어느 후보가 최선이었나

AUROC만 가장 높았던 후보는 `mix_ema_0.5`의 76.71%다. 하지만 Recall 29.75%가 직전 공통 경보 모델 30.83%보다 낮았다. 여러 평균 지표를 함께 유지·개선하려는 목적에는 `max_mix_0.1`을 균형 후보로 골랐다.

이 후보의 16장면 평균은 AUROC 73.41%, AP 54.73%, FPR 8.58%, Recall 32.08%, Precision 56.85%, F1 37.06%, Balanced accuracy 61.75%, Event coverage 84.21%다. 서로 다른 후보에서 지표별 최고값만 모아 만든 가상 모델 수치가 아니다.

## 이번 검토에서 새로 확인한 온라인 한계

정지 함수 `_motion`은 첫 lag개 값을 첫 계산 가능한 `result[lag]`로 채운다. lag4의 경우 5번째 특징을 바꾸면 이미 앞의 4개 motion도 바뀐다. 검토에서 앞의 특징을 모두 고정하고 5번째 특징만 바꿨을 때 초기 motion `[0,0,0,0]`이 `[1,1,1,1]`로 바뀌는 것을 재현했다.

이는 **라벨 누수는 아니지만 미래 관측 의존**이다. 현재 수치는 전체 특징 sequence를 계산한 탐색 실험으로 보존한다. 팀원 E17의 검증된 causal state와 새 로컬 stall 모듈의 검증 상태는 구분해야 한다. 63개 자동 테스트와 R01 115관측 bundle 실행 성공이 이 초기화 문제까지 검사했다는 뜻은 아니다.

엄격한 온라인 버전에서는 lag가 쌓일 때까지 해당 motion을 unavailable로 두고, 가능한 과거 정보만으로 계산해야 한다. 초기 fallback 정책·phase 추정·분포·q99.5·결합 q99를 동일하게 다시 보정하고 16장면을 재평가해야 한다. 기존 수치에 작은 코드 수정을 얹어 “같은 성능의 온라인 완성본”이라고 주장할 수 없다.

## 최종 후보 선택의 한계

학습/분포/임계값 계산에는 실제 테스트 라벨을 쓰지 않았지만, **54개 설정의 실제 테스트 성능을 비교한 후 후보를 골랐다.** 이것은 test-set model selection이다. 논문에는 exploratory 결과라고 공개해야 하며 독립 데이터나 새 촬영 원본 split에서 계수를 고정한 확인 실험이 필요하다.

R04와 S10의 AUROC가 낮아졌고 R01의 recall/F1도 일부 줄었다. “전체 평균을 개선하는 균형안”과 “모든 공정의 AUROC 하락이 없는 보장”은 전혀 다른 주장이다.
