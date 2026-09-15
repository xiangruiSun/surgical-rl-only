# SurgicAI Approach — RL deployment on a real dVRK PSM

> **Grasping and lifting now live in [`README_GRASP_LIFT.md`](README_GRASP_LIFT.md)**
> (`run_grasp_lift.py`): approach → close the jaw → observe → lift 1.5 cm, with
> a fail-closed precheck and an operator gate. This file still describes the
> approach-only path (`run_approach.py`), which is unchanged.


Runs the released `r6_unified_single_goal_yaw15_seed1_final.zip` checkpoint (or
the D2 servo, or a blend) as a closed loop against `PSM1/measured_cp`, using
nothing but a **start pose** (read from the arm) and a **goal position** (given
on the command line), both in the frame `measured_cp` reports — `ECM` on
lcsr-dvrk-15.

No AMBF, no perception stack, no training tree. The only inputs are two poses.

---

## Read this before you run anything

The checkpoint was decoded and its embedded demonstration set inspected
(`tools/inspect_checkpoint.py`). What the R6 policy was actually trained to do:

| | trained on |
|---|---|
| goal position (training frame) | x −3.30…−2.72 cm, y +1.95…+2.54 cm, z −12.23…−11.44 cm |
| distinct goals | 50, all inside a ±3 mm / ±15° box — this is a **single-goal** policy |
| travel per episode | median 4.1 cm |
| approach direction, tool frame | x −1.35…+3.21, y **+0.95…+3.95**, z +0.70…+4.42 cm |
| wrist rotation start → goal | 25.7°…100.2°, median **65°** |
| jaw | starts ≈0.76 open, goal 0.00 closed |
| arm | **PSM2** in simulation (`Approach_env.py` sets `psm_idx = 2`) |

Your task, with the goal orientation held at the current orientation:

* travel 3.16 cm — fine;
* approach direction in the tool frame `(−1.42, −0.61, +2.75)` cm — the y
  component is **outside** everything the policy ever saw, and x is at the edge;
* wrist rotation start → goal **0°** — the policy never once approached without
  rotating the wrist.

That last row is the decisive one. A rigid frame transform can move the numbers
into the training frame, but it cannot change the *relative* geometry between
start and goal, and the relative geometry is out of distribution.

### What that looks like in practice

`tools/offline_check.py` replays the loop against a perfect kinematic arm — the
optimistic case. On your exact numbers:

| controller | frame mode | outcome | closest approach | final rotation error |
|---|---|---|---|---|
| **RL** | rebase | never converges (200 steps) | 0.98 cm | 61.6° |
| **RL** | rebase + `trained_relative` | never converges | 0.85 cm | 120.1° |
| **RL** | translate | never converges | 1.34 cm | 77.4° |
| **RL** | identity (raw ECM) | never converges | 1.46 cm | 170.5° |
| **residual** (0.5·policy + 0.75·servo) | rebase | success, 12 steps | 0.94 cm | 1.2° |
| **D2 SE(3) servo** | — | success, 20 steps | **0.15 cm** | 0.0° |

The RL policy tumbles the wrist and orbits the goal. That is not a bug in this
code — the observation contract is verified byte-exact against 502 observations
stored inside the checkpoint (`tools/verify_contract.py`). It is the policy
doing what it was trained to do, on a task it was not trained for.

So: run the **dry run** first, look at the trace, and consider `--controller d2`
as the thing that actually reaches your goal. Your own `MODEL_ASSETS.md` already
says the default demo path is the staged D2 controller and RL is the
experimental alternative; that is consistent with what I measured.

---

## Install (on lcsr-dvrk-15)

```bash
cd ~/surgicai-rl-only
python3 -m venv --system-site-packages .venv-deploy
source .venv-deploy/bin/activate
pip install -r requirements-deploy.txt
```

`--system-site-packages` keeps `rclpy` and the message packages from
`/opt/ros/humble` visible. CPU torch is enough — the actor is a 3×256 MLP.

Verify the checkpoint and the observation contract:

```bash
source /opt/ros/humble/setup.bash
python3 tools/inspect_checkpoint.py --model models/rl/r6_unified_single_goal_yaw15_seed1_final.zip
python3 tools/verify_contract.py    --model models/rl/r6_unified_single_goal_yaw15_seed1_final.zip
# expect: PASS: observation contract reproduced exactly
```

## 1. Offline, no robot

```bash
python3 tools/offline_check.py \
  --model models/rl/r6_unified_single_goal_yaw15_seed1_final.zip \
  --start-pos  -0.05639860616831881 0.03366166453830251 0.024455994074878362 \
  --start-quat  0.23319925218484056 0.4267863636861243 -0.23588767438897446 0.841325450478807 \
  --goal-pos   -0.050726357 0.015332369 0.049514053 \
  --controller rl --frame-mode rebase --verbose
```

Add `--lag 0.3 --noise-mm 0.2` to model an arm that only closes 70 % of each
command and a noisy `measured_cp`. Swap `--controller d2` / `residual` to
compare. `--json-out run.json` saves the full trace.

## 2. Dry run on the robot

Publishes nothing. Reads `measured_cp`, freezes the goal, and logs the command
it *would* send at 10 Hz:

```bash
python3 run_approach.py \
  --model models/rl/r6_unified_single_goal_yaw15_seed1_final.zip \
  --goal-pos -0.050726357 0.015332369 0.049514053 \
  --controller rl --trace dryrun.jsonl
```

Check in the log that:

* `frame` is `ECM` — the goal you pass must be in the same frame as `measured_cp`;
* `start` matches what `ros2 topic echo /PSM1/measured_cp --once` shows;
* the commanded steps are ≤ 1.5 mm and stay inside the workspace box.

## 3. Move the arm

```bash
python3 run_approach.py \
  --model models/rl/r6_unified_single_goal_yaw15_seed1_final.zip \
  --goal-pos -0.050726357 0.015332369 0.049514053 \
  --controller d2 --interface move_cp --rate 2 --trace live.jsonl --execute
```

Start with `--controller d2 --interface move_cp --rate 2`. `servo_cp` streams
raw setpoints with no trajectory smoothing; `move_cp` interpolates and is more
forgiving for a first run. Keep a hand on the e-stop: the guards below are
software, and software guards do not stop a runaway arm.

---

## Safety guards (all active in `--execute`)

| guard | default | flag |
|---|---|---|
| workspace box around start+goal | +2 cm padding | `--workspace-pad-cm` |
| max translation per command | 2.5 mm | `--max-step-translation-mm` |
| max rotation per command | 5° | `--max-step-rotation-deg` |
| abort if the arm lags the command | 1.5 cm | `--max-tracking-error-cm` |
| abort if `measured_cp` goes stale | 0.25 s | — |
| episode cap | 200 steps | `--max-steps` |
| jaw | **frozen**, policy jaw output discarded | `--use-policy-jaw` to enable |

The jaw is frozen by default. Training drove it from open to closed during the
approach; on a real arm that would actuate the gripper mid-motion, so it is off
until you ask for it.

## Key options

| flag | meaning |
|---|---|
| `--controller rl \| d2 \| residual` | learned policy, SE(3) servo, or the guarded blend |
| `--frame-mode rebase \| translate \| identity` | how ECM-frame poses are bridged into the policy's training frame (see below) |
| `--goal-orientation hold \| trained_relative \| explicit` | `hold` keeps the current orientation (your choice); `trained_relative` rotates the wrist by the ~65° the policy expects; `explicit` takes `--goal-quat` |
| `--interface servo_cp \| move_cp` | direct setpoint stream, or interpolated moves |
| `--success-trans-cm` / `--success-rot-deg` | stopping tolerance, default 1 cm / 10° (the R6 evaluation contract) |

### Frame modes

The policy consumes *absolute* poses, and it learned them in the PSM base frame
around one frozen goal. Raw ECM numbers land nowhere near that goal, so every
absolute block of the 21-dim observation is off-distribution.

* `rebase` (default) — build `X = T_trained_goal · T_goal_ecm⁻¹` and run the
  policy in that frame, then map the commanded pose back. The `desired_goal` the
  network sees is then exactly the vector it was trained on. This is the
  relative-servo idea from your own R6 report §7.1.
* `translate` — position-only shift, orientation untouched. A/B baseline.
* `identity` — raw ECM values, for comparison only.

`d2` is frame-independent; the mode only affects `rl` and `residual`.

---

## Layout

```
run_approach.py                 ROS 2 entry point (approach only)
run_grasp_lift.py               ROS 2 entry point (approach + grasp + lift)
requirements-deploy.txt
surgicai_rl_deploy/
  contract.py                   frozen obs/action contract + measured training support
  frames.py                     SE(3) helpers and the frame bridge
  obs.py                        21-dim observation builder (verified against the zip)
  policy.py                     CPU-safe checkpoint loading + SHA256 identity check
  controllers.py                RL / D2 servo / residual blend
  loop.py                       the closed loop, safety clamps, success test
  ros_node.py                   topics, dry run, JSONL trace
  sequence.py                   grasp+lift phase machine  (README_GRASP_LIFT.md)
  jaw.py                        jaw units and grasp evidence          "
  plan.py                       start / grasp / lifted geometry       "
  feasibility.py                fail-closed precheck                  "
  grasp_lift_node.py            grasp+lift ROS node                   "
  mock.py                       kinematic arm + jaw model             "
tools/
  inspect_checkpoint.py         what the checkpoint contains and was trained on
  verify_contract.py            observation builder vs the checkpoint's own data
  offline_check.py              replay against a kinematic mock, no robot
  offline_grasp_lift.py         replay the whole grasp+lift sequence, no robot
  calibrate_jaw.py              what an empty jaw close looks like on your arm
tests/                          143 tests; no ROS, no robot, no checkpoint
```

## Contract notes

* Observation: `concat(achieved, desired, desired − achieved)`, positions in
  **cm**, orientation in **rad** as extrinsic-xyz RPY (KDL `GetRPY`, scipy
  `"xyz"`), jaw normalised 0…1.
* Action: `[-1, 1]^7`, applied as
  `cmd_raw = measured_raw + action · [1.5 mm, 1.5 mm, 1.5 mm, 3°, 3°, 3°, 0.05]`
  — translation in **metres** while the observation is in cm. That asymmetry is
  the training contract, not a bug.
* Loading the zip needs `custom_objects` overrides: the checkpoint pickles
  `RL_algo.DemoHerReplayBuffer` (absent outside the training tree) and CUDA
  tensors in `demo_data` (absent on a CPU host). `policy.py` handles both; the
  actor never touches either.
