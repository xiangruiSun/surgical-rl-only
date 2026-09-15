# Retraining SurgicAI for a wider workspace — what is worth doing, and what is not

Written after measuring the released checkpoints through the real deployment
loop. Every number below is reproducible with the tools named beside it.

---

## First: the approach policy's trained support is not what limits this task

A goal-conditioned policy's support, as read off its own demonstrations, is
expressed in two quantities:

```
tool offset   R_start^T (p_goal - p_start)
rotation      geodesic(R_start, R_goal)
```

Both are **invariant under any rigid transform**. So for any grasp pose there
exists a start pose that puts the episode inside the support, and it can be
computed directly rather than searched for (`deploy/surgicai_rl_deploy/staging.py`).

Measured over the ±3 mm / ±30° needle envelope this repository already assumes,
on the recorded lcsr-dvrk-15 geometry (`deploy/tools/workspace_spec.py`):

| | |
|---|---|
| in support **without** staging | 0 / 400 |
| in support **with** staging | **400 / 400** |
| staging poses span | 3.8 × 2.2 × 0.0 cm |
| travel from the parked pose | median 3.66 cm, max 4.14 cm |
| beyond the 8 cm path-radius guard | 0 / 400 |

**A 3.7 cm move covers the entire envelope.** Retraining the Approach policy
over a wider *support* would not change what the deployment can do. If someone
proposes a retrain to "extend the workspace", ask which of the three things
below they actually mean, because only those are real.

---

## What retraining genuinely buys

### 1. Robustness to the arm not being where it was told

The demonstrations embedded in the released Approach checkpoint contain **one
distinct start pose across all fifty episodes** (`tools/profile_checkpoint.py`
reports `unique_starts`). `Approach_env.reset` has the randomisation written
out and commented:

```python
# low_limits = [-0.02, -0.02, -0.01, -np.deg2rad(30), ...]
# random_array = np.random.uniform(low=low_limits, high=high_limits)
# self.psm_goal_list[self.psm_idx-1] = self.init_psm2+random_array
self.psm_goal_list[self.psm_idx-1] = np.copy(self.init_psm2)
```

(`Low_level_env_complete.reset` is worse: it *computes* `random_array_psm1/2`
and then never adds them.) This fork already threads
`psm_reset_random_range` through, defaulting to 5 mm and 0.5 rad per axis —
turning it on for a fresh Approach run is the cheapest real improvement
available, and it costs nothing outside the simulator.

This matters on hardware for a specific reason. Training's observation is the
integrated command, never the measured pose (`subtask_env.step` never reads
`measured_cp` back inside a subtask). The deployment now matches that, which is
worth 100% instead of 24% on an arm that closes 30% of the commanded gap per
cycle. But open-loop tracking only stays honest while the arm actually follows;
a policy that has seen start-pose spread is the one that survives when it does
not.

### 2. A wider needle envelope — **but only after perception**

`needle_reset_ranges.py` in this repository caps needle yaw at ±30° and says
why: the pose audit "was reliable at 20 degrees but developed near-180 degree
failures at 40 degrees". That is a perception limit, not a policy limit. A
policy trained over ±60° of needle yaw is correct about a goal nobody can
supply. **Widen the estimator first, re-run the audit, then widen the
curriculum's envelope** — `curriculum.approach_curriculum()` refuses to exceed
a stated envelope unless you pass a new one explicitly, so this cannot happen
by accident.

### 3. Place, which does not reproduce

Replaying each released checkpoint from its own demonstration starts, in the
training frame, against a perfect arm (`tools/replay_demos.py`):

| | at 0.5 cm / 30° (the env class default) | at the paper's tolerance | published |
|---|---|---|---|
| Approach | 25/25 | 21/25 at 1 mm / 10° | 96% ± 6% |
| Place | 22/25 | **12/25** at 5 mm / 10° | 97% ± 9% |

Approach reproduces. Place does not, and the gap is not the roll branch, not
the action scale, not the success metric — all three were swept. The likeliest
explanation is that the shipped `final_model.zip` is one seed while the
reported figure is a mean over five, but it is unexplained, and it is a
property of the file anyone would actually deploy.

Until that is resolved, the deployment carries the needle with the geometric
servo and runs Place as a logged-only shadow (`--shadow-model`). On the
recorded geometry the shadow reports itself out of distribution the moment the
transport begins and diverges 1.28 cm from the goal, while the servo arrives at
0.002 cm.

Separately: **Place is certified to thirty degrees of orientation error**
(every env class carries `threshold = [0.5, np.deg2rad(30)]`, and that is the
tolerance the published rates were measured at). For "angle the needle
correctly at the entry point", that is not a placement controller. Retraining
it against `TIGHT_THRESHOLD` is the only way it becomes one.

---

## Fix the training code before spending GPU time

| what | where | why |
|---|---|---|
| Two action scales | `Env_info/*` says 0.5 mm / 2°, `RL_training_online.py` says 1.0 mm / 3° | The first is the demonstration scale. Driving the released Approach checkpoint at it gives 7/20 instead of 19/20. `curriculum.training_env_kwargs()` returns the training one. |
| Roll branch | `subtask_env.Frame2Vec(bound=True)` | Roll lives in (−2π, 0], not scipy's ±π. Anything outside SurgicAI that re-derives RPY from a matrix must apply the same rule, or three of the twenty-one observation dimensions land on a branch no policy was trained on. |
| Success metric | `subtask_env.criteria()` | Uses ‖Δrpy‖, the Euclidean norm of the RPY difference vector, not the geodesic angle. They happen to agree closely here (swept: within one episode across every threshold pair), but the paper's prose says "orientation error" and means the second. Say which you are reporting. |
| Evaluation thresholds | `Model_evaluation.py` | Takes them from the command line and does not record them in the results file. Record them. Half the confusion in this document came from that. |

---

## The curriculum

`RL/curriculum.py`, unit-tested in `deploy/tests/test_curriculum.py` (26 tests,
no simulator needed).

**One dimension at a time.** Xie et al.'s analysis of randomisation effects on
sim2real ([arXiv:2206.06282](https://arxiv.org/pdf/2206.06282)) found that
all-at-once randomisation converged to a substantially lower return than the
unrandomised baseline, and that sequential strategies "seem to lead to a more
consistent real-world performance". Their range-finding heuristic — increase a
range until performance degrades by more than 10%, then stop — is implemented
as the regression rule: if widening costs more than `regress_margin` of success
against the best seen on that dimension, the curriculum freezes it at the last
width that worked and moves on.

**Order: start pose, then needle.** Start-pose spread is free; needle spread is
capped by perception.

```python
from curriculum import approach_curriculum, training_env_kwargs

curriculum = approach_curriculum()
env = gym.make("approach", **curriculum.env_kwargs(), **training_env_kwargs())

while not curriculum.done:
    model.learn(total_timesteps=EVAL_EVERY, reset_num_timesteps=False)
    rate = evaluate(model, env, episodes=20)
    verdict = curriculum.record(rate)
    print(verdict.reason)
    if verdict.changed:
        env = gym.make("approach", **curriculum.env_kwargs(), **training_env_kwargs())
```

Related literature, if you want to go further than a hand-ordered schedule:
[Automatic Goal Generation for RL Agents](https://arxiv.org/pdf/1705.06366)
(GoalGAN) generates goals at the frontier of current ability rather than on a
fixed schedule; [Diffusion-based Curriculum RL](https://proceedings.neurips.cc/paper_files/paper/2024/file/b0e89a49af1fb2ebea69bfc39df0be4a-Paper-Conference.pdf)
(NeurIPS 2024) is the current version of that idea;
[Flow-based Domain Randomization](https://arxiv.org/html/2502.01800) learns the
randomisation distribution instead of fixing it. All three are strictly more
machinery than a nine-dimension sequential schedule, and none of them addresses
the perception cap, which is the thing actually binding here.

---

## Acceptance gates

Do not ship a retrained checkpoint that has not cleared these.

1. `python3 deploy/tools/profile_checkpoint.py --model <new>` — roll outside
   (−2π, 0] must be **0%** on both desired and achieved, and `unique_starts`
   must be greater than 1.
2. `python3 deploy/tools/replay_demos.py --model <new> --compare` — the
   `surgicai_bound` / `obs=command` row must beat the published rate for the
   task, and `canonical` must be far worse (if it is not, the checkpoint was
   trained through a different convention and none of this applies to it).
3. `python3 deploy/tools/replay_demos.py --model <new> --arm-alpha 0.3` — a
   policy that only works against a perfect arm has not been made robust, it
   has been made lucky.
4. Register its SHA256 and profiled support in
   `deploy/surgicai_rl_deploy/contract.py`. A checkpoint with no contract now
   warns at load and the deployment refuses to stage into it.
5. `python3 deploy/tools/workspace_spec.py --contract <new>` on the real task
   geometry — confirm staging still covers the envelope. A wider support should
   make this easier, never harder.

## What not to do

- Do not pass `recover_step_size.py`'s output as `--trans-step-mm`. It fits the
  demonstrations, which is a different question from what the policy acts at.
- Do not widen the needle envelope past what the pose estimator has been
  audited at.
- Do not retrain Approach "for a wider workspace" without first reading the
  400/400 above and saying what it is you actually want that staging does not
  already give you.
- Do not train and evaluate at different step sizes without noticing. The
  released checkpoints did — `Env_info` and `RL_training_online.py` disagree —
  and it cost this project a fortnight.
