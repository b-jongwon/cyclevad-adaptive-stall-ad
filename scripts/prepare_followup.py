"""Create deterministic follow-up splits, annotation queue and experiment matrix.

This prepares a protocol only. It does not run experiments or fabricate labels.
"""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'configs/experiments/followup'
SCENES=['R01','R02','R03','R04']

def dump(name,obj):
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')

def ordered(ids,salt):
    return sorted(ids,key=lambda x:hashlib.sha256(f'{salt}|{x}'.encode()).hexdigest())

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    existing=OUT/'annotation_queue.json'
    if existing.exists():
        previous=json.loads(existing.read_text())
        assert all(x['status']=='unannotated' and all(x[k] is None for k in ['cycle_count','complete_cycles','anchors','ambiguous_intervals','process_anomaly_types','annotator','independent_reviewer']) for x in previous['items']), 'Refusing to overwrite started annotations; create a new version instead.'
    audit=json.loads((ROOT/'results/E0/data_audit.json').read_text())
    rows=audit['sequences']; by_id={r['id']:r for r in rows}
    folds={}; annotations=[]
    for scene in SCENES:
        normal=ordered([r['id'] for r in rows if r['scene']==scene and r['partition']=='training'],'followup-v1-20261009')
        folds[scene]=[]
        for fold in range(5):
            evaluation=normal[fold::5]
            remaining=ordered([x for x in normal if x not in evaluation],f'inner-v1-{fold}-42')
            n=max(2,round(len(remaining)*.15))
            parts={'validation':remaining[:n],'reference':remaining[n:2*n],'threshold':remaining[2*n:3*n],'fit':remaining[3*n:],'normal_evaluation':evaluation}
            assert all(parts.values())
            assert sum(map(len,parts.values()))==len(set(x for v in parts.values() for x in v))==len(normal)
            folds[scene].append({'fold':fold,'splits':parts})
        heldout=[x for f in folds[scene] for x in f['splits']['normal_evaluation']]
        assert sorted(heldout)==sorted(normal)
        pilot=folds[scene][0]['splits']
        selected=[(role,x) for role,count in [('fit',2),('validation',2),('normal_evaluation',4)] for x in ordered(pilot[role],'annotation-v1')[:count]]
        eligible=[r['id'] for r in rows if r['scene']==scene and r['partition']=='testing' and r['valid_frames']>0 and r['anomaly_frames']>0]
        selected += [('historical_anomaly',x) for x in ordered(eligible,'annotation-v1')[:2]]
        assert len(selected)==10
        for role,rid in selected:
            annotations.append({'id':rid,'scene':scene,'role':role,'fold':0,'frames':by_id[rid]['frames'],'status':'unannotated','cycle_count':None,'complete_cycles':None,'anchors':None,'ambiguous_intervals':None,'process_anomaly_types':None,'annotator':None,'independent_reviewer':None})
    manifest={'version':'followup-v1','status':'planned_not_run','seed':42,'unit':'video_file','recording_group_independence_verified':False,'note':'All derivatives of a source video stay in its fold. Replace file IDs with verified recording groups before confirmatory use.','scenes':folds}
    dump('normal_folds.json',manifest)
    dump('annotation_queue.json',{'version':'followup-v1','status':'unannotated','count':len(annotations),'items':annotations})
    training={'legacy':'max(r * exp(-d_min / (4 * temperature)), 0.1)','location_only':'max(r, 0.1)','uniform':'1'}
    scoring={'legacy':'r * exp(-d_min / (4 * temperature))','location_only':'r','none':'1'}
    cells=[{'id':f'C-{i}{j}','training_weight':a,'inference_gate':b,'training_formula':training[a],'inference_formula':scoring[b]} for i,a in enumerate(training) for j,b in enumerate(scoring)]
    protocol={
      'version':'followup-v1','status':'planned_not_run','base_config':'../replay_pinned.json','scenes':SCENES,'model_seed':42,'robustness_seeds':[43,44],
      'feature_policy':'Reuse verified frozen DINOv2 caches; recompute causal differences after temporal edits.',
      'historical_test_policy':'R01-R04 have already informed design: diagnostic only, not untouched confirmation.',
      'label_policy':'strict-v1: exclude complete mismatched-label videos, never trim or pad.',
      'timing':{'descriptor_lag_source_frames':4,'AR1_lag_source_frames':2,'progress_lags_source_frames':[2,8,32],'stride_values':[2,1],'unknown_fps':'Do not convert frames to seconds without evidence.'},
      'confidence':{'r':'abs(sum_j likelihood_j * exp(2*pi*i*theta_j))','not_a_probability':True,'training_weights':training,'inference_gates':scoring,'cells':cells,'readouts':['conditional_only','max(appearance,conditional)','max(appearance,conditional,process)'],'primary_diagnostic_contrasts':['C-00 vs C-01 vs C-02 (same trained model)','C-01 vs C-11 vs C-21 (same location-only inference gate)'],'candidate_fixed_before_followup':'C-21','untuned_control':'C-22','historical_winner_selection_forbidden':True,'hyperparameter_control':'Freeze tracker/theta, pooled/local model and C-00 normal-validation-selected Fourier order/ridge within each fold; refit mean and residual PCA only.','thresholds':'Recalibrate each readout using only reference and threshold splits.'},
      'tracker':{'variants':['T0_first_frame_initialized_clock','T1_framewise_template_match','T2_causal_filter'],'template_and_initial_observation_shared':True,'phase_origin':'Anchor coordinates and origin fixed by mapping manual FIT anchors to the shared training template; no equally spaced k/K assumption or test-time warping.','T1_control':'Same causal descriptor including the lag difference as T2; no recursive prior. This isolates filtering, not the existence of all temporal information.','primary_metrics':['anchor_circular_MAE_cycles','annotated_stage_interval_accuracy','transition_detection_delay_source_frames'],'secondary_metrics':['risk_coverage_using_confidence','recovery_delay','annotation_coverage'],'oracle_interpolated_phase_is_not_ground_truth':True,'pilot_annotation_videos':40,'expansion_rule':'If pilot is interpretable, annotate all remaining outer-fold-0 normal-evaluation videos before claiming tracking benefit.'},
      'process':{'score_arms':['appearance','appearance_plus_alignment','appearance_plus_innovation','appearance_plus_progress','appearance_plus_all_process','appearance_plus_AR1_transition'],'temporal_edits':['freeze','reverse','skip','swap_adjacent_blocks'],'severity_training_period_fractions':[0.05,0.10,0.20],'onsets_training_period_fractions':[0.25,0.50,0.75],'speed_stress_factors':[0.8,1.0,1.2],'speed_stress_normality':'Unverified: report stress alarm rate, not industrial normal FPR.','window_length_training_period_fraction':0.4,'detection_deadline_training_period_fraction':0.2,'state_warmup':'Run causally from sequence start; no reset at edit/window onset.','source_mapping_hidden_from_model':True,'AR1':'Ridge predictor across two source frames on same normal FIT PCA features; ridge from [0.01,0.1,1] using normal validation error.','exclusions':'Insufficient prefix/context/window is excluded for every arm; publish counts and reasons.','primary_metrics':['paired_fixed_window_AUROC','event_detection_rate_within_deadline','normal_evaluation_frame_FPR','normal_fixed_window_false_alarm_rate'],'secondary_metrics':['AP_with_exact_prevalence','conditional_detection_delay_and_missed_count','per_type_per_severity_metrics']},
      'uncertainty':{'bootstrap_resamples':2000,'seed':20261009,'unit':'source recording group when verified, otherwise source file','stratify_by':'scene','paired':True,'all_derivatives_resampled_together':True,'CI':'95% percentile interval','scope':'Descriptive exploratory uncertainty; file bootstrap does not prove recording independence; pilot may be inconclusive.'},
      'decision_rules':{'engineering_targets_not_literature_standards':True,'tracking':{'pilot_absolute_target_anchor_MAE_cycles':0.05,'relative_target_vs_T1_MAE_reduction':0.20,'required_scenes_for_point_target':3,'all_eligible_annotations_reported':True,'uncertainty_policy':'A broad paired CI means inconclusive, not success.'},'confidence':{'mechanism_check':'Gate intervention rescues conditional scores at fixed weights/angle; evaluate fitted-weight effect separately.','primary_readout':'max(appearance,conditional)','practical_target_macro_AP_gain_pp':1.0,'candidate':'C-21 compared with C-00','max_normal_evaluation_FPR_increase_pp':1.0,'caveat':'Historical-test gain is exploratory; no claim of solving calibration solely by meeting relative FPR target.'},'process':{'require_benefit_over':['appearance','appearance_plus_AR1_transition'],'practical_target_event_detection_gain_pp':5.0,'max_normal_evaluation_FPR':0.05,'normal_threshold_quantile':0.99,'caveat':'5% is a lenient screening ceiling, not desired deployment FPR. All scene values remain visible.'}},
      'stages':[{'id':'E6','status':'ready_to_implement','scope':'Normal fold audit + R01 calibration diagnosis + 40-video annotation packet'}, {'id':'E7','status':'requires_independent_annotations','scope':'Tracker T0/T1/T2 and confidence risk-coverage'}, {'id':'E8','status':'ready_to_implement','scope':'3x3 fit/inference confidence factorial; 3 readouts; same theta'}, {'id':'E9','status':'ready_to_implement_after_E7_diagnosis','scope':'Controlled process deviations, normal speed stress, source-frame-matched stride comparison'}, {'id':'E10','status':'requires_new_independent_recordings','scope':'Frozen candidate confirmation; no automatic best-test variant selection'}]
    }
    dump('protocol.json',protocol)
    summary={s:{'fold0_counts':{k:len(v) for k,v in folds[s][0]['splits'].items()},'normal_files':len([r for r in rows if r['scene']==s and r['partition']=='training'])} for s in SCENES}
    print(json.dumps({'confidence_cells':len(cells),'annotation_videos':len(annotations),'scenes':summary},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
