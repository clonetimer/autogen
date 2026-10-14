"""R72 robust, fail-closed *development-only* open-design acceptance rules.

Policy is fixed before seeing held-out evaluation. Scores are calculated on
TRAIN-derived environment-family leave-out folds, NOT FINAL-ID/FINAL-OOD.
"""
from dataclasses import dataclass
import math


class RobustDesignRejected(RuntimeError):
    def __init__(self, audit):
        self.audit = audit
        super().__init__('NO_CANDIDATE_MEETS_PREDECLARED_ROBUST_POLICY')


@dataclass(frozen=True)
class RobustDesignPolicy:
    overall_accuracy_min: float = .90
    worst_environment_accuracy_min: float = .80
    worst_fault_recall_min: float = .80
    healthy_false_alarm_max: float = .10
    unknown_max: float = .10
    min_environment_families: int = 3

    def __post_init__(self):
        for name, value in vars(self).items():
            if name == 'min_environment_families':
                if type(value) is not int or value < 3:
                    raise ValueError('INVALID_ROBUST_DESIGN_POLICY:' + name)
            elif (isinstance(value, bool) or not isinstance(value, (int, float)) or
                  not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError('INVALID_ROBUST_DESIGN_POLICY:' + name)

    def evaluate(self, metrics, classes, healthy_label):
        family_metrics = metrics.get('family_metrics')
        if not isinstance(family_metrics, dict) or len(family_metrics) < self.min_environment_families:
            return {'eligible': False, 'failed_gates': [{'metric':'min_environment_families'}]}
        failed = []
        def bound(name, observed, cutoff, op):
            if not isinstance(observed, (int,float)) or not math.isfinite(observed) or (
                observed < cutoff if op == 'min' else observed > cutoff):
                failed.append({'metric':name, 'value':observed,
                               'constraint':op, 'required':cutoff})
        total = metrics['total']
        bound('overall_accuracy', metrics['correct'] / total if total else None,
              self.overall_accuracy_min, 'min')
        bound('unknown_fraction', metrics['unknown'] / total if total else None,
              self.unknown_max, 'max')
        healthy_index = classes.index(healthy_label)
        families = {}
        for family, report in sorted(family_metrics.items()):
            count = report['total']
            health = report['healthy_count']
            if count == 0 or health == 0 or len(report['recall']) != len(classes) or not all(
                report['recall'][i] >= 0 and report['recall'][i] <= 1
                for i in range(len(classes))):
                failed.append({'metric':'INVALID_FAMILY_SUPPORT', 'family':family})
                continue
            # Every fault must be represented in each claimed evaluation family;
            # a 0 recall in an absent class does not provide adequate evidence.
            if any(int(report.get('class_support', {}).get(str(c), 0)) <= 0
                   for c in classes):
                failed.append({'metric':'MISSING_CLASS_SUPPORT', 'family':family})
            accuracy = report['correct'] / count
            far = report['healthy_nonacceptances'] / health
            min_fault_recall = min(r for i,r in enumerate(report['recall']) if i != healthy_index)
            bound('family_accuracy:' + family, accuracy, self.worst_environment_accuracy_min, 'min')
            bound('family_worst_fault_recall:' + family, min_fault_recall,
                  self.worst_fault_recall_min, 'min')
            bound('family_healthy_false_alarm:' + family, far, self.healthy_false_alarm_max, 'max')
            families[family] = {'accuracy':accuracy, 'worst_fault_recall':min_fault_recall,
                                'healthy_nonacceptance_rate':far}
        if not families:
            failed.append({'metric':'NO_VALID_FAMILY'})
        worst_miss = max((1-v['worst_fault_recall'] for v in families.values()), default=1.)
        worst_accuracy_loss = max((1-v['accuracy'] for v in families.values()), default=1.)
        score = .5 * worst_miss + .5 * worst_accuracy_loss
        return {'eligible':not failed,'failed_gates':failed,'per_environment':families,
                'robust_selection_loss':score,
                'qualification':'DEVELOPMENT_SELECTION_ONLY'}
