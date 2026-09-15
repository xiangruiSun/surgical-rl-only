# The suturing pipeline on a real dVRK PSM

**stage → approach → settle → close the jaw → observe → lift → transport →
place**, on real hardware, with no AMBF and no perception stack.

The task is described by three tool poses in the frame `measured_cp` reports:

| | |
|---|---|
| start | read from the arm; you do not pass it |
| grasp | where the **gripper** must be to take the needle |
| suture | where the **gripper** must be for the needle to sit correctly angled at the entry point |

All three are *tool* poses. After a blind grasp the needle-in-jaw transform is
unknown, so a needle-frame target could not be turned into a command, and this
deployment never pretends otherwise.

The phases are driven off the plan, so the same code runs a bare grasp-and-lift
(no suturing pose) or the full pipeline, and Insert or Pullout drop in as two
more segments whenever someone decides that is a safe thing to do.

```
run_pipeline.py              ROS 2 entry point for the whole pipeline
run_grasp_lift.py            the same node; grasp and lift only
run_approach.py              approach only, as before
surgicai_rl_deploy/
  contract.py                per-checkpoint action scale, tolerance, support
  sequence.py                the phase state machine (no ROS, no torch)
  staging.py                 solve a start pose inside a policy's own support
  jaw.py                     jaw radians <-> normalised jaw, and grasp *evidence*
  plan.py                    start / staged / grasp / lifted / via / suture
  feasibility.py             the fail-closed precheck
  grasp_lift_node.py         topics, dry run, operator gate, JSONL trace
  mock.py                    kinematic arm + jaw model for offline replay
  loop.py, obs.py, ...       the policy loop and observation builder
tools/
  calibrate_jaw.py           what an empty close looks like on YOUR arm
  offline_grasp_lift.py      rehearse the whole pipeline with no robot
  replay_demos.py            does a checkpoint work through THIS loop?
  profile_checkpoint.py      read a checkpoint's training support off its demos
  recover_step_size.py       the scale the demonstrations integrate at
  plan_r6_start.py           solve a staging pose (CLI over staging.py)
  sweep_r6_support.py        does the policy work anywhere in that support?
  fetch_upstream_checkpoints.sh   pull the released SurgicAI checkpoints
tests/                       272 tests, no ROS, robot or checkpoint required
```

---

## Read this first: there is no grasp sensor

In simulation, `SRC_approach.criteria()` calls
`psm.actuators[0].actuate("Needle")` and `grasp_status()["needle_grasped"]` is
ground truth: a magnetic constraint plus a finger ghost sensor. **Neither
exists on a real dVRK.** Nothing in this package can tell you the needle is in
the gripper.

What it can do is *report the jaw*, and it reports it as an observation:

| channel | what it means | how it lies |
|---|---|---|
| jaw angle residual | the jaw stopped further open than an empty jaw does | the dVRK jaw position comes from the motor through cables, so a blocked jaw partly shows as residual and partly as cable stretch; slack tendons and a mis-zeroed jaw fake it in both directions |
| jaw effort | the motor is straining against something | not every arm publishes `effort` on `jaw/measured_js`; friction drifts with the tool |

Every evidence record carries `grasp_verified: false`, every summary carries
it, and the final log line says it out loud. `--grasp-gate evidence` will act
on that evidence if you tell it to, and that choice is written into the trace.

Rehearsed offline against the jaw model, the honest picture is:

| case | evidence gate |
|---|---|
| empty gripper | refuses to lift |
| jaw stops at +3° (a 0.5 mm needle held near the pivot) | lifts |
| jaw stops at −14.5°, i.e. cable stretch hides nearly all the block | lifts, on the effort channel alone |
| same, with no `effort` field published | **refuses** — indistinguishable from empty |

So: if your arm publishes jaw effort, the evidence gate has two channels and is
worth something. If it does not, it has one weak channel, and `--grasp-gate
manual` — a human looking at the scene — is the only honest gate.

---

## The workspace question

The R6 checkpoint is a **single-goal** policy: 50 demonstrations inside a
±3 mm / ±15° box, median 65° of wrist rotation on the way in. The recorded task
sits outside that support on two tool axes and at 0° of wrist rotation, and the
offline replays show the policy orbiting the goal rather than reaching it.
Widening that region means retraining; a frame transform cannot change the
*relative* geometry between start and goal.

The geometric servo has no trained region at all. It works anywhere the arm can
physically go, which is why it is the default controller here. The workspace
question therefore stops being "is this inside the trained box" and becomes "is
this inside the reachable and safe box" — which is checkable up front, on
numbers, before anything moves. That is `feasibility.precheck`:

```
[  ok  ] inputs: approach 3.16 cm / 0.0 deg, lift 1.50 cm
[  ok  ] lift_direction: operator-confirmed: 1.50 cm along -z of the robot frame
[  ok  ] lift_vs_approach: the lift turns 143 deg away from the approach direction
[  ok  ] path_radius: every waypoint within 3.16 cm of the start (limit 8.0 cm)
[  ok  ] step_budget_approach: approach needs roughly 25 of 200 cycles
[  ok  ] lift_tolerance: success tolerance 0.20 cm is well inside the 1.50 cm lift
[  ok  ] grasp_gate: the arm will stop with the jaw closed and wait for a human
[  ok  ] r6_training_support: controller 'd2' is geometric, so the R6 trained
         region does not bound it (for reference, the RL support check would
         flag: tool-x -1.42 cm outside [-1.35, +3.21]; ...)
PRECHECK PASS
```

One `fail` and nothing is published. `--strict` makes every warning fatal.
The R6 support numbers are still printed under the servo, so you can see what
the RL path *would* have complained about and compare the two controllers on
the same real goal.

---

## The RPY branch defect (read this if you use `--controller rl`)

The released checkpoints diverged to ~130° of orientation error on every real
and simulated start, out of distribution or in. The cause was in this package,
not in the policies.

A rotation matrix is branch-invariant; the RPY triple describing it is not.
`scipy`'s `as_euler("xyz")` always returns roll and yaw in ±π, while the
SurgicAI environments integrate the RPY vector as free state and never
re-canonicalise it. In the upstream Approach checkpoint **100% of the
desired-goal rolls and 85% of the achieved-goal rolls lie outside ±π**:

```
training goal rpy      : [-3.774  0.497  1.317]
after matrix round-trip: [ 2.509  0.497  1.317]     difference: exactly 2π
```

Every cycle re-derived RPY from the measured rotation matrix, so three of the
twenty-one observation dimensions arrived on a branch no policy was trained on.

The fix is no longer a heuristic. `RL/subtask_env.py :: Frame2Vec(bound=True)`
says exactly what the branch is:

```python
roll, pitch, yaw = frame.M.GetRPY()
if roll <= np.deg2rad(-360):  roll += 2*np.pi
elif roll > np.deg2rad(0):    roll -= 2*np.pi
```

Roll lives in **(−2π, 0]**; pitch and yaw are left exactly as `GetRPY` returned
them. `frames.bound_roll()` is a transcription of that, and it reproduces
**50/50** stored goal rolls and **50/50** stored start rolls in the upstream
Approach checkpoint without being told the branch. Both released checkpoints
have 100% of their goal rolls outside ±π and 0% outside (−2π, 0].

`verify_contract.py` passed throughout because it rebuilds observations from the
stored 7-vectors and never round-trips a matrix — exactly the gap. It now checks
the round trip too. The D2 servo was never affected: it uses only *relative*
rotation, where a common 2π offset cancels.

## There are two action scales and it is easy to take the wrong one

`STEP_SIZE_RAW` was read out of the training sources and never verified. When
it finally was, the answer turned out to have two halves:

| source | Approach | what it is |
|---|---|---|
| `RL/Env_info/Approach_noise_env_info` | 0.5 mm / 2° | the scale the **demonstrations** integrate at |
| `RL/RL_training_online.py`, `RL/Model_evaluation.py` | **1.0 mm / 3°**, 300 steps | the scale the **policy** was trained and evaluated at |

`tools/recover_step_size.py` fits the first, exactly, to numerical zero — and
that is the wrong number to drive the policy with. Replaying the upstream
Approach checkpoint from its own demonstration starts, in the training frame,
against a perfect arm:

```
0.5 mm / 2 deg  (the demonstration scale)     7/20    35%
1.0 mm / 3 deg  (the training scale)         19/20    95%     published: 96% ± 6%
```

So use `tools/replay_demos.py`, which runs the policy, rather than a curve fit
against its training data:

```bash
python3 tools/replay_demos.py --model <checkpoint> --compare
```

## The observation was closed-loop and training's was not

`subtask_env.step` feeds the network `psm_goal_list` — the **integrated
command** — and never reads `measured_cp` back inside a subtask. The policy is
open-loop within an episode. This package fed it the measured pose.

A perfect arm hides the difference completely. A first-order arm does not:

| arm tracking per cycle | `obs=command` | `obs=measured` |
|---|---|---|
| 1.0 (perfect) | 25/25, 100% | 25/25, 100% |
| 0.5 | 25/25, 100% | 21/25, 84% |
| 0.3 | 25/25, 100% | **6/25, 24%** |

That is the hardware failure mode this project spent weeks chasing. The
observation now comes from an internal integrator seeded once from the staged
pose (`observation_source="command"`, the default); the measured pose is used
for the tracking-lag guard, the slip watch and the success test, and nothing
else. Geometric servo segments keep the closed-loop observation, which is what
a servo is for.

Because the integrator is seeded once and never re-derived from a rotation
matrix, the branch defect above cannot recur mid-episode either. The two fixes
are the same fix seen from two sides.

## What the published success rates were measured at

Every env class in SurgicAI carries `threshold = [0.5, np.deg2rad(30)]`, and
`Model_evaluation.py` takes the threshold from the command line without
recording what was passed. At **0.5 cm / 30°** this loop reproduces Approach
25/25 and Place 22/25, against published 96% ± 6% and 97% ± 9%. At
`Env_info`'s tighter 10° the same runs give 23/25 and 12/25.

Read that carefully before trusting a policy to place a needle: **Place is
certified to thirty degrees of orientation error.** It is not a precision
orientation controller, whatever "angle the needle correctly" needs it to be.
Both tolerances are recorded per contract (`success_*` and `env_info_*`).

---

## Putting the RL policy back in distribution

The R6 support is two constraints on the *relative* geometry between the start
pose and the grasp pose:

```
tool offset   R_startᵀ·(p_grasp − p_start)   in [−1.35, 0.95, 0.70] … [3.21, 3.95, 4.42] cm
rotation      geodesic(R_start, R_grasp)      in [25.7, 100.2] deg
```

Both are **invariant under the frame bridge** — a rigid transform cancels in
the tool offset and leaves a geodesic unchanged — so they can be satisfied by
choosing the start pose in the robot's own frame, and `--frame-mode rebase`
then puts the absolute goal on top of the trained one. With the grasp pose
fixed the solve is direct:

```
R_start = R_grasp · Rel⁻¹        Rel = the demonstrations' mean start→goal rotation
p_start = p_grasp − R_start · offset
```

`tools/plan_r6_start.py` does it and prints the three commands that follow:

```bash
python3 tools/plan_r6_start.py \
  --grasp-pos  <x y z> --grasp-quat <qx qy qz qw> \
  --current-pos <x y z> --current-quat <qx qy qz qw>
```

It reports where the solved pose sits in the box, the margin to each face, and
how far the arm must travel and turn to get there. Sitting at the
demonstrations' mean offset leaves more than 1 cm of margin on every axis and
31° on the rotation, so a millimetre of positioning error cannot push the
episode back out of support.

`tools/sweep_r6_support.py` answers the question properly, by sampling the box
rather than trusting one point: it grids the offsets, sweeps the rotation and
the jaw, verifies every start is in support, and runs both the policy and the
servo from each so the two are measured on identical geometry.

```bash
python3 tools/sweep_r6_support.py --model <checkpoint> \
  --grasp-pos <x y z> --grasp-quat <qx qy qz qw> --grid 3
```

A zero success rate across the box means the policy does not work inside the
region it was trained on, and no start-pose engineering will change that.

**Solving for an in-support start removes the out-of-distribution excuse. It
does not promise the policy works.** Run `tools/offline_check.py --controller rl --model ...` on the solved
pair before moving anything, and compare against `--controller d2` on the same
pair. If the policy still will not converge from an in-support start, the
geometry was never what was wrong with it — and that is a result worth having.

Two things the solve cannot tell you: whether the pose is **reachable**, and
whether the wrist can get there without sweeping through something. Reorienting
to the solved start typically means a large wrist rotation, because the support
*requires* the start and grasp orientations to differ by 25–100°. Move there
with the servo and a tight tolerance, watching the arm:

```bash
python3 run_approach.py --goal-pos <solved> --goal-quat <solved> \
  --goal-orientation explicit --controller d2 --interface move_cp --rate 2 \
  --success-trans-cm 0.2 --success-rot-deg 2.0 --execute
```

The demonstrations also started with the jaw at 0.76 normalised, which is 45.6°
on the default calibration; the jaw is part of the observation, so
`--jaw-approach-open-deg 45.6` is worth matching for the policy run. Close and
lift stay on the geometric servo regardless of what drives the approach.

---

## The sequence

| phase | what happens | how it ends |
|---|---|---|
| `stage` | servo to the pose that puts the approach policy inside its own demonstrated support. Skipped without `--stage`. | within tolerance of the staging pose; failure means the policy is never started at all |
| `approach` | the contract-verified `ApproachLoop` with `rl` / `d2` / `residual`, jaw held open | within tolerance of the grasp pose |
| `settle` | station-keep and require the measured pose to stop **drifting** | `settle_steps` clean cycles, or timeout → abort |
| `close` | ramp the jaw 3°/cycle to the squeeze angle, still station-keeping | ramp complete + dwell |
| `observe` | hold everything still and watch the jaw | `observe_steps` cycles, then the gate |
| `lift` | translate 1.5 cm, orientation frozen, jaw held squeezed | within 2 mm of the lift pose |
| `transport` | carry the needle to a point one lift-distance **above** the suturing pose, turning the wrist on the way. Never descends. Skipped without `--suture-pos`. | within tolerance of the via point |
| `place` | descend onto the suturing pose with the orientation already correct | within 2 mm / 3° of the suturing pose |
| `hold` | station-keep wherever the run ended, take a final reading | `hold_steps` cycles → done |

The transport goes over the top rather than straight there because a straight
line from the lift pose to an entry point can pass **below** the tissue plane in
the middle while carrying a needle. `--transport-via direct` is available and
the precheck warns about it. SurgicAI's own `Place_env.mid_goal_evaluator` uses
the same idea: a raised waypoint 3.5 cm before the entry.

Losing jaw evidence during the transport or the descent **stops and holds**; it
does not lower. Lowering is the right answer above the pickup point and the
wrong one halfway to the entry point.

If the approach policy does not converge, the default `--on-approach-failure
hold` stops at the last command with the jaw untouched and waits for a person.
`servo` finishes the approach geometrically and carries on, which is what an
unattended run wants and a first run does not.

The settle phase compares the mean pose over the last N cycles against the mean
over the N before it. A per-cycle motion test looks sensible and is wrong: on an
arm with a few tenths of a millimetre of noise on `measured_cp` it never falls
below any useful threshold, and a perfectly stationary arm times out. Raise
`--settle-translation-tol-mm` if you still see settle timeouts; the abort
message prints the measured drift so you can pick a number rather than guess.

**On abort the last command is held and the jaw is never opened.** A gripper
that may be holding a needle above the tissue does not get opened
automatically. Take manual control. The cycle that ends a run issues no fresh
command at all — the arm keeps tracking the pose it was already given.

## The shadow controller

`--shadow-model <checkpoint>` runs a second policy alongside the transport and
place legs, on the poses the arm actually visits, and **logs what it would have
commanded without ever publishing it**. That is how to answer "would the
SurgicAI Place policy have worked here" without letting it hold a needle.

On the recorded geometry for this task it answers plainly: the Place checkpoint
reports itself out of distribution the moment the transport begins —

```
tool-z +1.48 cm outside [-1.40, -1.17]
start->goal rotation 65.4 deg outside [94.0, 136.2]
```

— and then diverges into the clamp guard, 1.28 cm from the goal. The servo
carried the needle to 0.002 cm in the same run.

The shadow is fed the measured pose rather than its own integrator, because the
question is counterfactual: *from where the arm actually is, what would this
policy do next*. Had it been driving it would have visited different poses, so
the numbers are evidence, not a simulation. A shadow that throws is disabled and
recorded; it can never take down the run.

---

## Jaw units

dVRK jaw angles are radians, and **negative is squeeze** — `dvrk.psm`'s own
`jaw.close()` commands −20°, `jaw.open()` +60°. A command of 0.0 closes the
fingers but holds nothing. The defaults here:

| | angle | meaning |
|---|---|---|
| `--jaw-open-deg` | 60 | normalised 1.0 |
| `--jaw-approach-open-deg` | 40 | held during the approach |
| `--jaw-closed-deg` | 0 | normalised 0.0, fingers touching, no force |
| `--jaw-grip-deg` | −15 | the squeeze; floor is −25 |

The policy's observation uses the normalised 0…1 jaw, so the squeeze angle sits
*below* the normalised range by design. That is fine: the grip command is
produced by this package, not by the policy, whose jaw channel stays frozen
unless you pass `--use-policy-jaw` to the approach-only entry point.

With `--jaw-closed-deg 0` the mapping is `jaw_norm = jaw_rad / jaw_open_rad`,
which is exactly what the approach node already used — the approach path is
byte-identical.

---

## Running it

### 0. Calibrate the empty jaw (once per arm, redo after a tool change)

**The gripper must be empty.** The arm does not move in Cartesian space; only
the jaw is commanded.

```bash
source /opt/ros/humble/setup.bash
python3 tools/calibrate_jaw.py --arm /PSM1 --out jaw_baseline.json --execute
```

Without this file, jaw readings are logged raw with no reference, and
`--grasp-gate evidence` is refused by the precheck.

### 1. Offline, no robot

```bash
python3 tools/offline_grasp_lift.py \
  --start-pos  -0.05639860616831881 0.03366166453830251 0.024455994074878362 \
  --start-quat  0.23319925218484056 0.4267863636861243 -0.23588767438897446 0.841325450478807 \
  --grasp-pos  -0.050726357 0.015332369 0.049514053 \
  --suture-pos -0.040 0.005 0.040 --suture-quat 0 0 0 1 --suture-confirmed \
  --controller d2 --lift-sign -1 --grasp-gate evidence --verbose
```

Rehearse the failures too — `--empty-gripper`, `--drop-at-step N`,
`--no-jaw-effort`, `--lag 0.6 --noise-mm 0.3`, `--transport-via direct`. On the
recorded numbers the clean pipeline finishes in ~126 cycles and ends 0.002 cm
from the suturing pose.

To put the RL policy on the approach leg, add `--controller rl --model <zip>
--stage`; to measure the Place policy without letting it drive, add
`--shadow-model <Place zip>`.

### 2. Dry run on the robot

Publishes nothing. Reads `measured_cp` and `jaw/measured_js`, runs the
precheck, and logs every command it *would* send.

```bash
python3 run_pipeline.py \
  --grasp-pos  -0.050726357 0.015332369 0.049514053 \
  --suture-pos -0.040 0.005 0.040 --suture-quat 0 0 0 1 \
  --jaw-baseline jaw_baseline.json --trace dryrun.jsonl
```

Check in the log that `frame` is `ECM`, that `start` matches
`ros2 topic echo /PSM1/measured_cp --once`, that the lift target and the
suturing pose are where you expect, and that `jaw/measured_js` is actually
publishing (and whether it carries an `effort` field).

A dry run publishes nothing, so the arm cannot move. By default the dry run
therefore **simulates** a perfect arm landing on each command and walks the
whole sequence; poses after the first cycle are marked as simulated and are not
measurements. `--dry-run-static` keeps reading the real `measured_cp` instead,
in which case the approach can never converge and the episode always ends at
`max_steps` — that is the arm not moving, not the controller failing.

### QoS

Subscriptions default to **BEST_EFFORT** (`--sub-reliability`). A RELIABLE
subscriber does not match a BEST_EFFORT publisher, and dVRK state topics are
not uniform: on lcsr-dvrk-15 `measured_cp` matched a RELIABLE subscriber and
`jaw/measured_js` did not, so the jaw looked absent while `ros2 topic echo`
showed it publishing fine. BEST_EFFORT matches either kind. Command publishers
stay RELIABLE.

### 3. Rehearse with an empty gripper

Same command plus `--execute --grasp-gate always --controller d2 --interface
move_cp --rate 2`, with **nothing** under the gripper. This exercises the real
motion, the real jaw commands and the real lift with nothing to drop.

### 4. Live, with a human on the gate

```bash
python3 run_pipeline.py \
  --grasp-pos  -0.050726357 0.015332369 0.049514053 \
  --suture-pos -0.040 0.005 0.040 --suture-quat 0 0 0 1 --suture-confirmed \
  --controller d2 --interface move_cp --rate 2 \
  --lift-sign -1 --jaw-baseline jaw_baseline.json \
  --grasp-gate manual --trace live.jsonl --execute
```

The arm approaches, closes the jaw, stops, prints what it saw, and waits. Type
`y` + Enter to lift or `n` + Enter to stop — or publish on `--confirm-topic`.
If it lifts, it then carries the needle over to the suturing point and descends.

Two flags are **required** for `--execute` and the precheck refuses a live run
without them:

- `--lift-sign`: `+z` in the ECM frame is not guaranteed to be away from the
  tissue, and a wrong sign drives the needle into the pad.
- `--suture-confirmed`, whenever `--suture-pos` is given: that pose is where the
  **gripper** goes. Where the needle ends up also depends on how it sits in the
  jaws, which nothing in this package measures. Somebody has to have looked at
  the scene.

The most reliable way to get a suturing pose is to teach it: jog the arm by
hand until the needle sits correctly at the entry point, read
`ros2 topic echo /PSM1/measured_cp --once`, and pass that back. Then the pose
is a measurement rather than an estimate, and the needle-in-jaw transform it
implicitly encodes is the one you actually have.

Start with `--interface move_cp --rate 2`; `servo_cp` streams raw setpoints
with no trajectory smoothing. Keep a hand on the e-stop — everything here is
software, and software guards do not stop a runaway arm.

---

## Safety envelope

| guard | default | flag |
|---|---|---|
| precheck must pass | fail-closed | `--strict` to promote warnings |
| lift sign stated by a human | required for `--execute` | `--lift-sign` |
| workspace box around all three waypoints | +2 cm padding | `--workspace-pad-cm` |
| operator hard box | off | `--limit-low` / `--limit-high` |
| path radius from the measured start | 8 cm | `--max-path-radius-cm` |
| max translation per command | 2.5 mm | `--max-step-translation-mm` |
| max rotation per command | 5° | `--max-step-rotation-deg` |
| abort if the arm lags the command | 1.5 cm | `--max-tracking-error-cm` |
| abort if `measured_cp` goes stale | 0.25 s | — |
| abort if `jaw/measured_js` goes stale | 0.5 s | `--max-jaw-age-s` |
| jaw squeeze floor | −25° | `--jaw-grip-deg` |
| lift ceiling | 5 cm | `--lift-distance-cm` |
| needle lost during the lift | abort | `--on-slip abort\|continue\|lower` |
| jaw opened on abort | **never** | — |

Tested: every command in every offline episode stays inside the padded box and
under both per-step caps, and the jaw command never passes the grip angle.

---

## Gate modes

| `--grasp-gate` | behaviour |
|---|---|
| `manual` (default) | close, stop, wait for a human. The only honest gate. |
| `evidence` | lift if the jaw stayed blocked for N cycles. Needs a baseline. Records that you chose this. |
| `always` | lift regardless. For dry runs and empty-gripper rehearsals. |
| `never` | stop after the close. Useful for tuning the approach alone. |

---

## Simulation counterpart

`src/SurgicAI/RL/GraspLift_env.py` drives **the same** `GraspLiftSequencer` in
AMBF, importing it from this package rather than reimplementing it. The phase
logic, jaw ramp, settle test, slip monitor and safety clamps are one
implementation shared by both worlds; only the arm differs.

The one thing simulation has that hardware does not is ground truth, and it is
fed in through the sequencer's ordinary operator-confirmation hook: in AMBF
"the operator" is the finger ghost sensor. The same `manual` gate that waits
for a human on hardware waits for the simulator there. Each episode records the
ground-truth grasp state *and* the jaw evidence the real deployment would have
had, so you can measure directly how much the real run is flying blind.

```bash
source /opt/ros/humble/setup.bash
source "$HOME/ambf_ros_ws/install/setup.bash"
python3 RL/run_grasp_lift_sim.py --episodes 10 --trans_error 0.5 --angle_error 30
```

In simulation the lift defaults to **+z** in the PSM base frame: the reset pose
sits at z ≈ −0.08 and the needle goal at z ≈ −0.12, so +z is away from the pad.
This is the opposite sign from the ECM-frame example above, which is exactly
why the real run makes you state it.

---

## Tests

```bash
python3 -m pytest tests -q        # 272 tests, no ROS, robot or checkpoint
```

Covers the jaw mapping and evidence logic, the plan geometry including the via
point, the staging solver and its frame invariance, every precheck branch,
every phase transition and abort path, the safety envelope, the shadow
controller (including one that throws), the sim/real bridge, both command
lines, and the three contract defects:

- `bound_roll` against a transcription of `Frame2Vec(bound=True)`
- the acting scale kept distinct from the demonstration scale
- the open-loop integrator, pinned against a stuck arm that must not be able to
  move the policy's belief about where the tool is

plus the older regression guards on the approach contract, so none of this can
quietly change the approach.

The numbers quoted throughout this document come from `tools/replay_demos.py`
against the released checkpoints; reproduce them with

```bash
bash tools/fetch_upstream_checkpoints.sh
python3 tools/replay_demos.py --model ../models/rl/upstream/approach_td3_her_bc.zip --compare
python3 tools/replay_demos.py --model ../models/rl/upstream/approach_td3_her_bc.zip --arm-alpha 0.3 --compare
```
