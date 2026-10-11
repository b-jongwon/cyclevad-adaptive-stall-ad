# 출처와 변경 범위

기본 구현 및 E0–E17 실험 결과는 [PigeonLabs/KNU_Capstone1_CycleVAD](https://github.com/PigeonLabs/KNU_Capstone1_CycleVAD)에서 가져왔다.

- 확인한 upstream main commit: `075e21d0e49287f6d849ee70fc1074d6b0501b60`.
- 로컬 확장 원본의 마지막 학습 commit: `ddf0f95`.
- 기본 features/tracker/model 및 기존 paper/streaming 모듈과 tests의 출처는 upstream이다.
- 로컬 추가 실험: adaptive_final, universal_alarm, feature_study, ranking_fusion, 관련 tests/reports/configs.
- 새 상세 설명, 비교표, 공개 준비 및 초기 미래 참조 검토는 이번 정리에서 작성했다.

기존 연구와 팀원의 구현을 인용 없이 새 알고리즘으로 주장하지 않는다. upstream의 공개 여부 자체가 임의 재라이선스 권한을 뜻하지 않는다. 확인된 라이선스 파일이 없어 전체 코드를 MIT 등으로 재라이선스하지 않았다. 외부 재사용·상업 배포 조건은 원 저작자에게 확인해야 한다. IPAD 데이터셋 및 DINOv2 가중치의 조건도 각 배포처에 별도로 적용된다.

UPSTREAM_README.md는 upstream 원문 스냅샷, UPSTREAM_README_WITH_LOCAL_HISTORY.md는 로컬에서 개선 문단을 붙였던 과거 README다. authoritative 숫자 출처는 각 results 폴더의 JSON이다. 과거 README의 완전 인과적 처리 문장은 로컬 motion 초기화 검토 결과로 한계가 드러났으므로 그대로 확정 주장으로 사용하지 않는다.

## 공개 범위

코드, 설정, 숫자 결과, 테스트, 설명 문서를 공개한다. IPAD 이미지·영상·ZIP, feature cache, joblib/pt checkpoint, API 키·GitHub 자격 증명은 포함하지 않는다. 학습 모델은 사용자 로컬 실행 폴더에 보존한다.

이번 작업은 저장된 결과를 검토·정리·공개한 작업이며 전체 모델을 새로 재학습한 작업이 아니다. 최종 후보는 테스트 비교 후 고른 탐색 결과다. 알려진 motion 시작부 미래 관측 의존을 수정하고 정상 보정과 독립 평가를 다시 하기 전에는 운영용 완성 모델로 주장하지 않는다.
