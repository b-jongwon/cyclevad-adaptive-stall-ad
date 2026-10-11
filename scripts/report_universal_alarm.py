"""Generate a transparent Korean report from the all-scene experiment JSON."""
import argparse
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    macro = read(args.results / "macro_metrics.json")
    scene = read(args.results / "metrics_by_scene.json")
    selection = read(args.results / "validation_selection.json")
    params = read(args.results / "selected_rule.json")
    run = read(args.results / "execution.json")
    lines = ["# 전체 16개 장면 공통 경보 규칙 실험", "",
             "IPAD R01–R04, S01–S12 전체에서 동일한 알고리즘과 동일한 하이퍼파라미터를 사용했다. "
             "장면별 정상 분포와 임계값만 해당 장면의 정상 영상으로 추정한다. 새로운 공정에 정상 영상 없이 바로 적용되는 단일 범용 모델을 검증한 실험은 아니다.", "",
             "## 결과", "", "| 지표 (16개 장면 macro) | 기존 | 공통 규칙 | 차이 |",
             "| --- | ---: | ---: | ---: |"]
    for key, name in [("auroc", "AUROC"), ("ap", "AP"), ("fpr", "FPR ↓"),
                      ("recall", "Recall ↑"), ("precision", "Precision ↑"), ("f1", "F1 ↑"),
                      ("specificity", "Specificity ↑"), ("balanced_accuracy", "Balanced accuracy ↑"),
                      ("event_coverage", "Event coverage ↑")]:
        a, b = macro["baseline"][key], macro["universal"][key]
        lines.append(f"| {name} | {a*100:.2f}% | {b*100:.2f}% | {(b-a)*100:+.2f}%p |")
    lines += ["", "AUROC와 AP는 16개 장면 모두 기존 결과와 정확히 같다. "
              "이는 점수 배열을 그대로 유지한 결과이며, 정지 분기의 탐지 능력이 AUROC로 평가되었다는 의미는 아니다. "
              "최종 출력은 원래 연속 이상 점수와 별도의 경보 결정 두 가지이다.", "",
              "## 장면별 결과", "",
              "| 장면 | AUROC (동일) | 기존 FPR | 개선 FPR | 기존 Recall | 개선 Recall | 기존 F1 | 개선 F1 | 기존 event coverage | 개선 event coverage |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, values in scene.items():
        a, b = values["baseline"], values["universal"]
        numbers = [a["auroc"], a["fpr"], b["fpr"], a["recall"], b["recall"],
                   a["f1"], b["f1"], a["event_coverage"], b["event_coverage"]]
        lines.append("| " + name + " | " + " | ".join(f"{v*100:.2f}%" if v is not None else "—" for v in numbers) + " |")
    lines += ["", "## 정상 검증 데이터에서 선택한 공통 설정", "",
              "```json", json.dumps(params, indent=2), "```", "",
              "공통 후보 24개를 비교했다: 임계값 보정 계수 0/0.1/0.2/0.3, 연속 이상 관측 1/2/3개, 정지 분기 사용/미사용. "
              "실제 테스트 이상 라벨은 설정 선택에 사용하지 않았다.", "",
              "- 정상 reference: 영상 시작부 24개 관측의 낮은 20% 점수로 기준 강도 추정.",
              "- 정상 threshold holdout: 기존 임계값과 정지 점수 q99.5 임계값 추정.",
              "- 정상 validation: 오탐률과 특징 반복·구간 역전 합성 이상 탐지로 24개 후보 비교.",
              "- 선택 조건: 정상 validation macro FPR 증가 0.5%p 이하, 장면별 증가 2%p 이하, 합성 이상 frame recall 감소 1%p 이하.",
              "- 선택 목적: 합성 event coverage + 합성 frame recall − 5 × 정상 FPR 최대화.",
              "- 테스트: 공통 설정 저장 후 실제 테스트 라벨을 읽고 모든 지표 평가.", "",
              "## 온라인 처리와 성능 보존", "",
              "초기 23개 관측에는 기존 임계값을 사용한다. 24번째 관측부터 초기 강도 차이로 임계값을 보정하고 고정한다. "
              "미래 관측을 참조하지 않는 인과적 처리이다. 정지 분기는 lag 1·4 cosine motion의 phase별 lower-tail 점수, EMA, 연속 3개 관측 경보를 사용한다. "
              "지속 시간은 sampled observation 단위이며 원본 영상 stride 2를 유지한다.", "",
              "## 검증 선택 결과", "",
              "| 항목 | 기존 규칙 | 선택 규칙 |", "| --- | ---: | ---: |"]
    for key in ("normal_fpr", "synthetic_frame_recall", "synthetic_event_coverage"):
        lines.append(f"| {key} | {selection['baseline'][key]*100:.2f}% | {selection['chosen'][key]*100:.2f}% |")
    lines += ["", "합성 이상 결과는 실제 정지·역행 이상에 대한 성능 입증을 대체하지 않는다. "
              "정상 holdout 분포에서 안전한 설정도 실제 새 영상에서는 FPR이나 Recall이 악화될 수 있다. "
              "장면별 결과를 모두 공개하고 모든 장면에서 모든 경보 지표가 개선됐다고 주장하지 않는다.", "",
              "## 재현 및 저장 파일", "",
              "```powershell", "$env:PYTHONPATH='src'",
              "python -m cycle_vad.universal_alarm --data-root 'D:\\종프 학습\\IPAD_dataset' --run-root '<기존 모델·특징 캐시 폴더>' --output '<새 결과 폴더>'",
              "```", "",
              "- `selected_rule.json`: 전체 공정 공통 설정", "- `validation_selection.json`: 후보 24개와 정상 검증 결과",
              "- `metrics_by_scene.json`: 16개 장면 전체 지표", "- `macro_metrics.json`: 장면별 동일 가중치 평균",
              "- `per_video.json`: 모든 테스트 영상 지표, 미지 라벨 수, 경보 시작 횟수",
              "- `score_invariance.json`: AUROC/AP 보존 확인", "- `*_protocol.json`: 장면별 정상 데이터 분할",
              "- `*_alarm_calibration.joblib`: 장면별 경보 보정 모델", "",
              f"실행 시간: {run['seconds']:.1f}초 (특징 추출 및 미완료 5개 장면 기본 모델 학습 제외). "
              "DINOv2 특징은 기존 캐시를 재사용했다. CycleVAD와 경보 분포 추정은 epoch 반복을 사용하는 신경망 학습이 아니다. "
              "추가 유료 API 호출은 없었다.", "",
              "R01–R04는 이전 실험에서 이미 결과를 확인한 개발 장면이다. 이번 공통 규칙은 모든 장면의 정상 validation만으로 선택했으며 "
              "S01–S12의 실제 이상 라벨로 규칙을 재조정하지 않았다. 원본 recording 그룹 정보가 없어 정상 분할의 촬영 원본 독립성은 검증되지 않았다."]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
