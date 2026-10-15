"""R78 predeclared SYNTHETIC spurious-shift development stress benchmark.

This is not Basilisk and cannot qualify a diagnostic model. No target samples,
target moments, target class labels or metrics are used for fitting, early
stopping, candidate ranking or hyperparameter selection. The target is
generated only after all models have been fitted, once per frozen seed.
Rerunning after changing code turns this test into iterative DEV feedback.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from phm_design.task_candidate_registry import TorchTaskEstimator
from phm_design.neural_domain_generalization import (
    DGSequenceEstimator, DGTrainingPolicy,
    RiskExtrapolationEstimator, RiskExtrapolationPolicy,
)

SEEDS = (781, 782)
SEQUENCE_SAMPLES = 48
TRAIN_ENVIRONMENTS = (('source-a', 1.0, 0.0),
                      ('source-b', .8, .3),
                      ('source-c', .55, -.3))
TARGET_ENVIRONMENT = ('unseen-synthetic-shift', -.9, .7)
SAMPLES_PER_LABEL = 14
TRAINING_STEPS = 36  # optimizer steps, not identical FLOPs across algorithms


def environment(seed: int, *, strength: float, offset: float, nominal_noise: float = .14):
    rng = np.random.default_rng(seed)
    t = np.linspace(0., 1., SEQUENCE_SAMPLES, dtype=np.float32)
    x, y = [], []
    for label in (0, 1):
        signed_label = 2 * label - 1
        for replicate in range(SAMPLES_PER_LABEL):
            # Stable label mechanism is a *weak* temporal oscillation.
            stable = signed_label * .16 * np.sin(6 * np.pi * t)
            stable += rng.normal(0., nominal_noise, len(t))
            # A much larger nuisance correlates with training label; its
            # sign reverses in the isolated target environment.
            nuisance = offset + signed_label * strength + rng.normal(0., .12, len(t))
            nuisance += .04 * t
            # Gain drift is independent of true label.
            gain = rng.normal(1., .03)
            x.append(np.stack((gain * stable, nuisance), axis=-1))
            y.append(label)
    return np.asarray(x,dtype=np.float32), np.asarray(y,dtype=np.int64)


def train_only(seed: int):
    samples, labels, group_names = [], [], []
    for number, (group, strength, offset) in enumerate(TRAIN_ENVIRONMENTS):
        xx, yy = environment(seed*100+number, strength=strength, offset=offset)
        samples.append(xx)
        labels.append(yy)
        group_names.extend([group]*len(yy))
    return (np.concatenate(samples), np.concatenate(labels), np.asarray(group_names))


def _models(seed: int):
    return {
      'ERM-TCN': TorchTaskEstimator('torch_tcn', seed, epochs=6, batch_size=16),
      'GroupDRO-TCN': DGSequenceEstimator('dg_tcn', seed,policy=DGTrainingPolicy(
            epochs=12, steps_per_epoch=3)),
      'V-REx-TCN': RiskExtrapolationEstimator('rex_tcn', seed,policy=RiskExtrapolationPolicy(
            epochs=12, steps_per_epoch=3, risk_variance_warmup_epochs=2)),
    }


def _metrics(truth, probabilities):
    predicted = probabilities.argmax(axis=1)
    return {
        'accuracy': float(np.mean(predicted == truth)),
        'healthy_false_alarm': float(np.mean(predicted[truth==0] != 0)),
        'fault_recall': float(np.mean(predicted[truth==1] == 1)),
        'count': int(len(truth)),
        'probabilities_finite': bool(np.isfinite(probabilities).all()),
    }


def main():
    results = []
    # The target is generated only after all three fitting calls for this seed.
    for seed in SEEDS:
        x_train, y_train, groups = train_only(seed)
        trained = {}
        for model_name, estimator in _models(seed).items():
            if model_name == 'ERM-TCN':
                estimator.fit(x_train, y_train)
            else:
                estimator.fit(x_train, y_train, domain_groups=groups)
            trained[model_name] = estimator
        x_target, y_target = environment(seed*100+997,
                                        strength=TARGET_ENVIRONMENT[1],
                                        offset=TARGET_ENVIRONMENT[2])
        for name, estimator in trained.items():
            pred = estimator.predict_proba(x_target)
            results.append({'seed':seed, 'candidate':name,
                            'heldout':_metrics(y_target, pred),
                            'training_objective':estimator.training_.get('objective','empirical_risk'),
                            'trained_source_environment_count':len(TRAIN_ENVIRONMENTS),
                            'target_used_for_training_or_search':False,
                            'evaluation_role':'REVEALED_SYNTHETIC_DEV_STRESS'})
    summary = {}
    for name in ('ERM-TCN','GroupDRO-TCN','V-REx-TCN'):
        entries = [r['heldout'] for r in results if r['candidate']==name]
        summary[name] = {key+'_mean':float(np.mean([x[key] for x in entries]))
                         for key in ('accuracy','healthy_false_alarm','fault_recall')}
    contract = {'seeds':SEEDS,'train_families':TRAIN_ENVIRONMENTS,
                'target_family':TARGET_ENVIRONMENT,
                'sequence_samples':SEQUENCE_SAMPLES,
                'samples_per_label':SAMPLES_PER_LABEL,
                'training_steps_approx':TRAINING_STEPS,
                'model_selection_uses_target':False,
                'all_models_fit_before_target_generation_per_seed':True}
    digest = hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    report = {'schema':'phm.r78.synthetic-dev-stress.v1',
              'contract':contract,'contract_sha256':digest,
              'model_selection_uses_target':False,
              'evaluation_role':'REVEALED_SYNTHETIC_DEV_STRESS',
              'no_independent_qualification':True,
              'training_compute_not_identical':True,
              'individual':results,'summary':summary}
    output=Path('r78_synthetic_dev_stress.json')
    output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps({'contract_sha256':digest,'summary':summary,
                      'evaluation_role':report['evaluation_role'],
                      'no_independent_qualification':True},sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
