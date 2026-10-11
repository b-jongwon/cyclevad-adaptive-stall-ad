# E7 독립 cycle·anchor 주석 안내

E7의 위치 추적 정확도는 실제 주석이 있어야 측정할 수 있습니다. 자동 추적 결과나 영상 길이를 정답으로 삼지 않습니다. 현재 주석 완료 수는 **0개**입니다.

## 준비된 자료

- 대상: [고정된 40개 영상 목록](../configs/experiments/followup/annotation_queue.json). 장면별 FIT 2개, validation 2개, normal evaluation 4개, historical test 2개입니다.
- 로컬 뷰어: `runs/annotation_packet/index.html`. 영상 선택, 프레임 슬라이더, 좌우 방향키, 접촉시트를 제공합니다.
- 생성: 저장소 루트에서 `PYTHONPATH=src python scripts/prepare_annotation_packet.py` 실행. 현재 스크립트는 저장소 옆 `IPAD_dataset`을 읽습니다.
- 모델의 위치·신뢰도·이상 점수는 뷰어에 표시하지 않습니다. 접촉시트는 탐색용이며 전환 경계는 전후 프레임으로 확인해야 합니다.
- 원본 이미지와 접촉시트는 로컬에 보관하며 GitHub에 공개하지 않습니다.

## 기록 방식

프레임 인덱스는 0부터 시작합니다. 구간은 `[시작, 끝-exclusive]`로 기록합니다. 초 단위로 환산하지 않습니다.

1. 장면의 FIT 영상 두 개에서 반복 동작을 확인하고, 눈으로 구별할 수 있는 전환의 이름을 먼저 정합니다. 예: 집기 시작, 접촉, 이탈 등 실제 영상에서 확인한 이름을 사용합니다. 장면마다 다른 사전을 사용할 수 있습니다.
2. `cycle_count`에 확인 가능한 cycle 수를, `complete_cycles`에 완전한 cycle의 프레임 구간 목록을 기록합니다. 불완전하거나 경계가 보이지 않는 반복을 강제로 완전한 cycle로 만들지 않습니다.
3. `anchors`에 `{"frame": 프레임번호, "name": 전환이름}` 항목을 기록합니다. 전환 이름과 의미는 해당 장면의 FIT 영상에서 정한 사전을 유지합니다. 확정하기 어려운 전환은 `ambiguous_intervals`에 구간과 이유를 기록합니다.
4. `process_anomaly_types`에는 직접 확인 가능한 정지·역행·생략 등과 구간을 기록합니다. 확인 불가능하면 미확인으로 남기며 기존 이진 이상 라벨에서 유형을 추정하지 않습니다.
5. `annotator`에는 실제 주석 작성자를 기록합니다. 다른 검토자가 원본 프레임을 보고 확인한 경우에만 `independent_reviewer`를 채웁니다. 검토가 끝나기 전에는 `status`를 완료로 바꾸지 않습니다.

프레임 번호를 영상 길이로 나눈 값을 위치 정답으로 쓰거나, anchor를 균등한 k/K 위치에 배치하지 않습니다. Template상의 anchor 좌표와 phase 원점은 FIT 주석만으로 고정합니다. 평가 영상에 맞춘 사후 offset/DTW 정렬을 허용하지 않습니다.

## 다음 실행 조건

현재 T0(첫 관측으로 초기화한 clock), T1(프레임별 template 관측), T2(인과적 filter) 예측은 `runs/followup/Rxx/fold0/tracker/`에 저장되어 있습니다. 이는 주석이 아닌 모델 출력입니다.

독립 주석과 FIT anchor 사전이 준비되면 anchor circular MAE, 주석된 단계 구간 정확도, 전환 지연, 신뢰도별 risk–coverage를 계산합니다. 모호한 구간과 주석 커버리지를 함께 보고합니다. 그 진단을 바탕으로 E9의 진행 이상 실험을 시작하며, E10은 별도의 신규 촬영 자료가 필요합니다.
