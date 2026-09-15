# Grasp and lift on a real dVRK PSM

Extends the approach deployment to the full **approach → close the jaw →
observe → lift 1.5 cm** sequence, on real hardware, with no AMBF and no
perception stack. The only inputs are still two poses: a start pose read from
the arm and a grasp position given on the command line.

```
run_grasp_lift.py            ROS 2 entry point for the whole sequence
run_approach.py              unchanged: approach only, as before
surgicai_rl_deploy/
  sequence.py                the phase state machine (no ROS, no torch)
  jaw.py                     jaw radians <-> normalised jaw, and grasp *evidence*
  plan.py                    start / grasp / lifted geometry
  feasibility.py             the fail-closed precheck
  grasp_lift_node.py         topics, dry run, operator gate, JSONL trace
  mock.py                    kinematic arm + jaw model for offline replay
  loop.py, obs.py, ...       unchanged from the approach deployment
tools/
  calibrate_jaw.py           what an empty close looks like on YOUR arm
  offline_grasp_lift.py      replay the whole sequence with no robot
tests/                       143 tests, no ROS or robot required
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

## The sequence

| phase | what happens | how it ends |
|---|---|---|
| `approach` | the existing, contract-verified `ApproachLoop` with `rl` / `d2` / `residual`, jaw held open | within tolerance of the grasp pose |
| `settle` | station-keep and require the measured pose to stop **drifting** | `settle_steps` clean cycles, or timeout → abort |
| `close` | ramp the jaw 3°/cycle to the squeeze angle, still station-keeping | ramp complete + dwell |
| `observe` | hold everything still and watch the jaw | `observe_steps` cycles, then the gate |
| `lift` | translate 1.5 cm, orientation frozen, jaw held squeezed | within 2 mm of the lift pose |
| `hold` | station-keep, take a final reading | `hold_steps` cycles → done |

The settle phase compares the mean pose over the last N cycles against the mean
over the N before it. A per-cycle motion test looks sensible and is wrong: on an
arm with a few tenths of a millimetre of noise on `measured_cp` it never falls
below any useful threshold, and a perfectly stationary arm times out. Raise
`--settle-translation-tol-mm` if you still see settle timeouts; the abort
message prints the measured drift so you can pick a number rather than guess.

**On abort the last command is held and the jaw is never opened.** A gripper
that may be holding a needle 1.5 cm above the tissue does not get opened
automatically. Take manual control.

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
  --controller d2 --lift-sign -1 --grasp-gate evidence --verbose
```

Rehearse the failures too — `--empty-gripper`, `--drop-at-step N`,
`--no-jaw-effort`, `--lag 0.6 --noise-mm 0.3`. On the recorded numbers the
clean run finishes in ~89 cycles and ends 1.50 cm above the grasp pose.

### 2. Dry run on the robot

Publishes nothing. Reads `measured_cp` and `jaw/measured_js`, runs the
precheck, and logs every command it *would* send.

```bash
python3 run_grasp_lift.py \
  --grasp-pos -0.050726357 0.015332369 0.049514053 \
  --jaw-baseline jaw_baseline.json --trace dryrun.jsonl
```

Check in the log that `frame` is `ECM`, that `start` matches
`ros2 topic echo /PSM1/measured_cp --once`, that the lift target is where you
expect, and that `jaw/measured_js` is actually publishing (and whether it
carries an `effort` field).

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
python3 run_grasp_lift.py \
  --grasp-pos -0.050726357 0.015332369 0.049514053 \
  --controller d2 --interface move_cp --rate 2 \
  --lift-sign -1 --jaw-baseline jaw_baseline.json \
  --grasp-gate manual --trace live.jsonl --execute
```

The arm approaches, closes the jaw, stops, prints what it saw, and waits. Type
`y` + Enter to lift or `n` + Enter to stop — or publish on `--confirm-topic`.

`--lift-sign` is **required** for `--execute`: the precheck refuses a live run
without it. `+z` in the ECM frame is not guaranteed to be away from the tissue,
and a wrong sign drives the needle into the pad. Check it in the scene.

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
python3 -m pytest tests -q        # 143 tests, no ROS and no robot
```

Covers the jaw mapping and evidence logic, the lift geometry, every precheck
branch, every phase transition and abort path, the safety envelope, the
sim/real bridge, and regression guards on the approach contract
(`STEP_SIZE_RAW`, the 21-dim observation layout, `cmd = measured + action *
step`, and the recorded D2 result) so that adding grasp and lift cannot quietly
change the approach.
