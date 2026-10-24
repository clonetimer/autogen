# R83h — Actual 45 physical-group Basilisk TRAIN excitation and observability QA

**Date:** 2026-10-24. **Research status:** physics-source characterization complete;
NO new machine-learning model fitted, NO VALIDATION/TEST/FINAL opened, NO
independent diagnostic performance or flight/equipment qualification claimed.

## Immutable provenance

- Original real Basilisk 126-case source experiment:
  [R83f Actions #40099391872](https://github.com/clonetimer/autogen/actions/runs/40099391872).
  Original canonical source-plan SHA-256:
  \`c54bfcd154322a620c7d8058ce8a40db6e155602593c5ff5da8bdba43ef3b714\`.
- Original 45 TRAIN nominal/fault observability audit SHA-256:
  \`2784ad179972b63cd1b43baaa404553202f0be3c5794d8589b8531d522471b8e\`.
  Original source-plan audit SHA-256:
  \`bc7754da78fc05921f5dd1cfce9052e6310310a3dad0b6aa011e7277a75425ac\`.
- [Actual physics-only QA CI #40105231844](https://github.com/clonetimer/autogen/actions/runs/40105231844):
  **PASS** Linux Python 3.11, Linux Python 3.12, Windows Python 3.11
  synthetic-contract regressions, then real original R83f source QA.
- [Original R83h source-linked QA artifact](https://github.com/clonetimer/autogen/actions/runs/40105231844):
  \`r83h-real-basilisk-train-45-pairs-physics-excitation\`;
  retrieved ZIP SHA-256:
  \`fb7a29204802d2e213b84c9cf498d43e6e4567c9838a383ef46f880b41befbb2\`.
  Includes full 45-pair measurements and 9 regime/mechanism summaries.
- Source reader opens only 45 original TRAIN normal/fault groups (90 original CSV
  traces), matching each original SHA-256, TaskSpec parameters, random seed,
  Basilisk backend/source version, fault start time and affected wheel.
  **ZERO** VAL, TEST, FINAL files are opened. Results are descriptive,
  retrospective, and must never be fed into model selection or inference.

## Predefined descriptive physical metrics

For every source original pair, select the **actual injected wheel for
physics QA ONLY**, and split the original two matched traces by their
recorded fault-onset time. Calculate:
- Nominal (healthy-twin) *post-event* mean absolute RW commanded torque (N*m).
- The ratio of that post-event mean absolute commanded torque to the
  **nominal trace's reported effective maximum torque** at corresponding
  timestamps; this is a nominal loading proxy, **not a true measured
  electrical current or delivered motor torque**.
- Post-event maximum absolute paired wheel-speed difference (rad/s).
- Post-event mean absolute difference between commanded and **simulator
  engineering applied torque** under nominal and faulty conditions. Applied
  torque is source-internal protected physics truth, never a deployable
  algorithm input channel.
- Per physical fault-class (15 pairs) and mechanism × inclination-region (5
  pairs) medians/minima/maxima. No uncertainty or causal effect is inferred
  from 5/15 small, non-independent-source-family groups.

## Measured source results

| Physical fault mechanism | Independent source pairs | Median nominal mean absolute command (N*m) | Median nominal mean command / rating | Median max post-fault wheel-speed difference (rad/s) | Median fault command-applied gap (N*m) |
|---|---:|---:|---:|---:|---:|
| RW friction increase | 15 | 0.00647739 | 3.1174% | 3.761270 | 0.122895 |
| RW jam | 15 | 0.00909734 | 3.9058% | 26.835564 | 0.734003 |
| RW torque-authority loss | 15 | 0.00404457 | **2.5565%** | **0.307233** | **0.007408** |

The torque-authority-loss normal twin command-to-rating median is small
across all three preregistered inclinations:

| Inclination regime | Physical pairs | Median nominal command/rated capability | Median speed delta (rad/s) | Median fault command-applied gap (N*m) |
|---|---:|---:|---:|---:|
| 29° (low) | 5 | 2.5565% | 0.307233 | 0.007408 |
| 47° (medium) | 5 | 3.4808% | 0.411820 | 0.011060 |
| 65° (high) | 5 | 1.9027% | 0.185570 | 0.005209 |

The normal-source pre-event target-wheel speed discrepancy was exactly zero
for every audited physical pair. The original 45-pair physics receipt's
post-event maximum speed differences were recomputed and matched without
tolerance adjustment.

### Scientific interpretation

The **data are consistent with under-excitation / insufficient command
variation as a possible limitation** of the RW authority-loss source
dataset. In this specific Basilisk simulator family the commanded control
torque is small relative to the nominal limit; the fault-related speed
effect is also much smaller than for jam and friction. However the command
ratio, fault response and source mechanisms are not randomized, so these
statistics **do not prove under-excitation caused classifier failure**.
Not every actuator capacity loss is necessarily observable through
closed-loop speed at low command. Causal identification needs a prospective
experiment with controlled command histories.

It would be scientifically invalid to relabel the 60 R83f TRAIN records as
60 independent physical groups (there were 45 groups and 15 paired twins),
use source-internal applied torque as an inference channel merely because
it is visible in the simulator trace, relax worst-regime/class recall or
healthy-FAR gates, or evaluate the hidden R83f VALIDATION cohort just to
select an excitation profile.

### R83i preregistration recommendation (NOT executed here)

Before generating any new physical data, independently review the native
FSW/RW actuator command interface and supported action scheduling in the
pinned SatAgentSimulation source. Register at least two actuator excitation
regimes — original closed-loop reference and a safe, bounded nonzero
command excitation — with identical event/severity distribution, physical
pair seeds, groups, and same balanced fault-class quota. Predeclare the
actuator limit, angle/rate safety bounds, excitation duration, waveform,
source allocation and early stopping criteria before any labels/results are
seen. Test actual Basilisk wheel applied torque and speed response and
maintain command-vs-applied distinction.

Once true physics manifests an adequately identifiable loss mechanism,
rerun the same fixed/static/adaptive TRAIN comparison under frozen budgets
and then use a NEW prospective blind cohort for independent confirmation.
Do not overwrite the current **R83f and R83g negative studies**.

**R83h determination:** physics-source observability QA PASS; possible
insufficient command excitation merits prospective testing; diagnostic
model eligibility remains REFUSED (unchanged R83f/R83g).
