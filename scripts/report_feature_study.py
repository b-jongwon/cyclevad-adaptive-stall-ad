"""Publish all positive and negative ablations without hiding scene regressions."""
import argparse
import json
from pathlib import Path


def load(p):
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--fusion", type=Path, required=True)
    p.add_argument("--previous", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    features, fusion = load(args.features / "macro_metrics.json"), load(args.fusion / "macro_metrics.json")
    scenes = load(args.fusion / "metrics_by_scene.json")
    previous = load(args.previous / "macro_metrics.json")
    baseline, selected, best_rank = fusion["baseline"], fusion["max_mix_0.1"], fusion["mix_ema_0.5"]
    keys = ("auroc", "ap", "fpr", "recall", "precision", "f1", "event_coverage", "balanced_accuracy")
    lines = ["# 전체 16개 공정: 특징 학습·시간 모델·점수 결합 고도화", "",
             "## 최종 균형안", "",
             "최종 균형 후보는 `max_mix_0.1`이다. R01–R04, S01–S12 모두 동일한 파이프라인과 결합 계수를 사용했다. "
             "각 공정의 정상 분포는 해당 공정 정상 영상으로 추정한다. 특정 장면의 테스트 결과를 보고 장면마다 모델을 교체하지 않았다.", "",
             "| 지표 (16개 장면 동일 가중치 평균) | 최초 고정 임계값 | 직전 공통 경보 모델 | 이번 균형 후보 |",
             "| --- | ---: | ---: | ---: |"]
    for k in keys:
        initial = previous["baseline"].get(k)
        lines.append(f"| {k} | " + (f"{initial*100:.2f}%" if initial is not None else "—")
                     + f" | {baseline[k]*100:.2f}% | {selected[k]*100:.2f}% |")
    lines += ["", "## 구체적인 파이프라인", "",
              "1. 기존 동결 DINOv2 특징 추출과 정상 학습 CycleVAD 점수 계산.",
              "2. lag 1·4 global cosine motion을 계산해 정상 phase별 lower-tail 움직임 부족 점수 추정.",
              "3. 정지 점수 EMA(0.5) 평활화.",
              "4. 각 분기의 정상 threshold holdout 기준으로 강도를 맞춤.",
              "5. `최종 연속 점수 = max(CycleVAD 점수 / 정상 임계값, 1.1 × 정지 점수 / 정상 정지 임계값)`.",
              "6. 최종 점수의 정상 holdout q99로 기본 임계값 추정.",
              "7. 초기 24개 관측의 낮은 20% 강도로 영상별 임계값 0.3 보정. 기존 공통 정지 경보 분기 병행.",
              "8. 연속 점수와 프레임 경보를 출력.", "",
              "이번에는 경보만 바꾼 것이 아니라 연속 이상 점수 자체에 정지 정보를 반영했다. "
              "따라서 AUROC/AP도 바뀐다. 표의 AUROC/AP와 경보 지표는 같은 `max_mix_0.1` 후보의 결과이며 "
              "서로 다른 후보의 최고 지표를 조합하지 않았다.", "",
              "## 추가 학습 내용", "",
              "- 각 공정 정상 fit에서 global memory 1,024개, position별 patch memory 512개, motion memory 256개 학습.",
              "- 정상 fit 특징으로 PCA 64차원 변환과 scale 추정.",
              "- 16개 공정 각각 정상 특징 denoising autoencoder 학습: 최대 30 epoch, validation patience 4.",
              "- 16개 공정 각각 과거 8개 관측으로 현재 특징을 예측하는 GRU 학습: 최대 30 epoch, validation patience 4.",
              "- 이번 실행에서는 모든 AE·GRU가 30 epoch까지 학습되었다. 가장 낮은 정상 validation loss checkpoint를 복원했다.",
              "- 모든 neural head와 memory는 정상 fit만 학습하고, reference/threshold/validation 영상과 실제 테스트 이상은 fit에 포함하지 않았다.",
              "- 사전학습 DINOv2 encoder 가중치는 갱신하지 않았다. 추가 유료 LLM/VLM API 호출 없음.", "",
              "AE·GRU나 메모리 확장을 단순 추가한 설정은 이 데이터에서 기존 모델보다 낮은 AUROC를 보였다. "
              "최종 균형 후보에는 이 신경망 head를 억지로 넣지 않았다. 학습 checkpoint와 모든 실패 결과도 보존했다.", "",
              "## 추가로 비교한 전체 설정", "",
              "총 54개 고유 설정을 비교했다. 특징 head 16개 설정과 시간 점수 결합 39개 설정에서 중복 baseline 1개를 제외한 수치다. "
              "이들은 54개의 서로 다른 신경망이 아니라 모듈·결합 비율·시간 평활화·지속 시간 조건의 실험 설정이다.", "",
              "| 실험 | AUROC | AP | FPR ↓ | Recall | Precision | F1 | Event coverage | Balanced accuracy |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for prefix, results in [("features", features), ("fusion", fusion)]:
        for method, values in results.items():
            lines.append("| " + prefix + "/" + method + " | " + " | ".join(f"{values[k]*100:.2f}%" for k in keys) + " |")
    lines += ["", "## 지표별 절충", "",
              f"AUROC 최대 후보 `mix_ema_0.5`는 {best_rank['auroc']*100:.2f}%를 기록했다. "
              f"그러나 Recall {best_rank['recall']*100:.2f}%로 직전 모델 {baseline['recall']*100:.2f}%보다 낮아 "
              "모든 평균 지표 개선이라는 요구에는 `max_mix_0.1`이 더 맞는다.", "",
              "## 장면별 전체 결과와 예외", "",
              "| 장면 | 이전 AUROC | 이번 AUROC | 이전 AP | 이번 AP | 이전 FPR | 이번 FPR | 이전 Recall | 이번 Recall | 이전 F1 | 이번 F1 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    degraded = []
    for name, r in scenes.items():
        a, b = r["baseline"], r["max_mix_0.1"]
        vals = [v for k in ("auroc", "ap", "fpr", "recall", "f1") for v in (a[k], b[k])]
        lines.append("| " + name + " | " + " | ".join(f"{v*100:.2f}%" for v in vals) + " |")
        if b["auroc"] < a["auroc"]:
            degraded.append(name)
    lines += ["", "전체 평균 주요 지표는 모두 좋아졌지만, 모든 장면의 모든 지표가 동시에 좋아진 것은 아니다. "
              "AUROC가 하락한 장면: " + ", ".join(degraded) + ". "
              "특정 공정에서 AUROC 감소를 절대 허용할 수 없다면 기존 모델을 유지해야 한다. "
              "정상 데이터만으로 미지의 실제 이상에서 모든 지표 상승을 보장할 수는 없다.", "",
              "## 실험 해석과 논문 검증", "",
              "실제 테스트 라벨은 학습·reference 보정·threshold 추정에 사용하지 않았다. "
              "하지만 이번 추천 후보는 여러 설정의 실제 테스트 결과를 비교하고 고른 탐색 결과다. "
              "따라서 독립 테스트에서 확인된 최종 일반화 성능처럼 주장하지 않는다. "
              "논문에는 후보와 계수를 고정한 뒤 새 데이터·새 정상/이상 recording split 또는 외부 데이터에서 재평가해야 한다. "
              "기존 테스트를 다시 나누는 것만으로 이전에 결과를 본 영향을 지울 수 없다.", "",
              "이번 실험이 뒷받침하는 주장은 정상 특징의 새 neural head를 늘리는 것보다, "
              "기존 score가 놓친 움직임 부족 정보를 정상 강도로 보정해 함께 읽는 것이 이 16개 장면의 평균 구분력과 경보 품질을 개선했다는 것이다.", "",
              "## 실행 가능한 저장물", "",
              "- `cyclevad_ranking_duration_all16/<scene>/balanced_pipeline.joblib`: 정상 모델, motion 분포, 결합 설정, 임계값을 담은 실제 inference bundle.",
              "- `cycle_vad.ranking_fusion.predict_fusion(data, bundle)`: 특징 cache를 받아 score, alarm, dynamic threshold, indices 출력.",
              "- `cyclevad_feature_study_all16/<scene>/models/heads.pt`: 추가 학습 AE·GRU와 정상 memory checkpoint.",
              "- `training.json`: 전체 epoch의 train/validation loss.",
              "- `metrics_by_scene.json`, `macro_metrics.json`, `per_video.json`: 모든 결과."]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.fusion / "recommended_recipe.json").write_text(json.dumps({
        "method": "max_mix_0.1", "shared_across_all_scenes": True,
        "selection_status": "exploratory after real test comparison",
        "all_macro_metrics_improved_vs_previous": True,
        "per_scene_auroc_regressions": degraded, "metrics": selected,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
