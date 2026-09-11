# SurgicAI Suturing Policy Sim2Real

本仓库把 SurgicAI Approach 阶段整理成一个可检查、可替换输入、失败可定位的主入口：

```text
AMBF 场景
  └─ ECM RGB ──> Depth Anything (DA) ──> metric depth
       mask + RGB + depth ──> FoundationPose (FP) ──> 首帧物理门
       └─ AMBF GT pose（仅用于对照）                 │
                                                    ▼
                                           frozen needle goal
                                                    │
                        PSM measured pose ──────────┤
                                                    ▼
                                        D2 controller / RL policy
                                                    │
                                                    ▼
                                                  Reach
```

当前发布验证到 **Reach**；没有声称真实 dVRK 的自动 mask、hand-eye 精度、闭爪、提针已经完成。完整边界见 [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md)。

真机候选筛选现已拆成可比较的接口：针可选 `fp_only / flat / support / flat_planar / support_planar`，PSM 可选 `disabled / vision_only / fk_only / fused`。所有缺失标定均 fail-closed，模式、topic、JSON schema 与待测实参见 [`docs/REAL_PERCEPTION_MODE_SWITCHES.md`](docs/REAL_PERCEPTION_MODE_SWITCHES.md)。

## 1. 先理解目录结构

### 四个外部依赖（不放进本仓库）

| 名称 | 作用 | 推荐位置 |
|---|---|---|
| AMBF | 启动物理仿真和 ROS 场景 | `$HOME/surgicai_external/ambf` |
| SRC | Surgical Robotics Challenge 的 ADF、world、针 mesh | `$HOME/surgicai_external/surgical_robotics_challenge` |
| DA | 从 RGB 预测 metric depth | `$HOME/surgicai_external/Depth-Anything-V2` |
| FP | 用 RGB + depth + mask + CAD 求针的 6D 位姿 | `$HOME/surgicai_external/FoundationPose` |

此外还需要 AMBF ROS 2 bridge（推荐 `$HOME/ambf_ros_ws`）和外置 DA checkpoint（推荐 `$HOME/surgicai_models/p5a_vitl_fp32_best.pth`）。

### 本仓库的主模块

```text
configs/pipeline.env.example   一处填写所有本机路径
scripts/doctor.py              只检查依赖，不启动仿真/机器人
scripts/prepare_ambf_launch.py 把 SRC 本机路径写入 AMBF launch
scripts/run_simulation.py      唯一仿真入口：code / S3 / S4 / full
scripts/inspect_results.py     检查一次运行是否真的完整
src/perception/                RGB、DA、FP、首帧物理门
src/control/                   frozen goal、D2 控制器、Reach 评估
src/SurgicAI/RL/              Approach 环境和 TD3-HER-BC
src/runners/                   被主入口调用的 S3/S4 runner
ros2_ws/src/suturing_runtime/  真机 ROS topic 适配与受保护 Reach 网关
models/rl/                     已发布的 M3/R6 checkpoint
data/reference/                不重跑 DA/FP 时使用的冻结 S3 bank
```

新手只需要按下面顺序操作。不要直接运行 `src/runners/*.sh`，也不要从历史实验报告里复制零散命令。

### 真机图片先分段验收，不要一上来跑主程序

拿到真实 ECM 首帧后，先用 `scripts/run_real_perception_stage.py` 把故障边界拆开：

```text
inspect（图片可读/曝光）
  -> da（checkpoint 身份 + 单张深度）
  -> mask（人工折线的多宽度敏感性）
  -> bundle（RGB/depth/mask/K/shape 对账）
  -> run_fp_stage_docker.sh（独立 FoundationPose register）
  -> fp-compare（K、mask 改动是否令位姿跳变）
```

这些命令全部是只读离线检查，不启动 AMBF、不发 ROS 运动命令。运行
`python scripts/run_real_perception_stage.py --help`查看每一段的参数；真机候选合同见
[`docs/REAL_PERCEPTION_MODE_SWITCHES.md`](docs/REAL_PERCEPTION_MODE_SWITCHES.md)。程序成功退出只说明API/文件合同成立；没有真实深度与6D位姿参考时，不能把它写成DA/FP准确率通过。

## 2. 设置 AMBF、SRC、DA、FP 路径

推荐在 Ubuntu 22.04 / WSL2 Ubuntu 22.04、ROS 2 Humble、Python 3.10 下运行，并把所有目录放在 Linux home，避免 OneDrive 路径中的空格。

```bash
cd "$HOME"
git clone https://github.com/18244241528jm-cpu/suturing-policy-sim2real.git
cd suturing-policy-sim2real

cp configs/pipeline.env.example configs/pipeline.env
nano configs/pipeline.env
```

若你采用推荐目录，下面五项不需要改；否则只改等号右边：

```bash
AMBF_ROOT=${HOME}/surgicai_external/ambf
SRC_ROOT=${HOME}/surgicai_external/surgical_robotics_challenge
DA_ROOT=${HOME}/surgicai_external/Depth-Anything-V2
FOUNDATIONPOSE_ROOT=${HOME}/surgicai_external/FoundationPose
DA_CHECKPOINT=${HOME}/surgicai_models/p5a_vitl_fp32_best.pth
```

再检查 ROS 与输出路径：

```bash
SIM_S3_ROS_SETUP=/opt/ros/humble/setup.bash
SIM_S3_AMBF_ROS_SETUP=${HOME}/ambf_ros_ws/install/setup.bash
SURGICAI_RESULT_ROOT=${HOME}/surgicai_runs
ROS_DOMAIN_ID=220
```

`ROS_DOMAIN_ID` 必须是当前无人使用的 `1..232`。同一 domain 上残留的 AMBF/ROS 节点会污染结果。

把配置加载到当前 shell：

```bash
set -a
source configs/pipeline.env
set +a
```

立即核对路径，不要等到运行数小时后才发现填错：

```bash
test -x "$AMBF_ROOT/core/build/bin/ambf_simulator" && echo "AMBF OK"
test -f "$SRC_ROOT/ADF/Phantoms/3D_MED/high_res/Needle_stage_d_v0.OBJ" && echo "SRC OK"
test -d "$DA_ROOT" && echo "DA repo OK"
test -d "$FOUNDATIONPOSE_ROOT" && echo "FP repo OK"
test -f "$DA_CHECKPOINT" && echo "DA checkpoint OK"
```

缺少外部源码时，可先建立推荐目录并 clone；AMBF、DA 和 FP 仍需按各自上游 README 完成编译/镜像安装：

```bash
mkdir -p "$HOME/surgicai_external" "$HOME/surgicai_models" "$HOME/surgicai_runs"
git clone https://github.com/WPI-AIM/ambf.git "$HOME/surgicai_external/ambf"
git clone https://github.com/surgical-robotics-ai/surgical_robotics_challenge.git \
  "$HOME/surgicai_external/surgical_robotics_challenge"
git clone https://github.com/DepthAnything/Depth-Anything-V2.git \
  "$HOME/surgicai_external/Depth-Anything-V2"
git clone https://github.com/NVlabs/FoundationPose.git \
  "$HOME/surgicai_external/FoundationPose"
git -C "$HOME/surgicai_external/FoundationPose" checkout \
  a1b694b83e633c2cb6115b9063d940a687759392
```

DA checkpoint 约 3.8 GB，不在 Git 中。向项目成员取得后必须校验：

```bash
sha256sum "$DA_CHECKPOINT"
# 期望：fc46bead4a5ea0e4122566bb88b93932aa82f110ee98281b5fcb09f499c9ec88
```

完整 FP 路径还要求 GPU container 的本地标签存在：

```bash
docker image inspect foundationpose:blackwell >/dev/null && echo "FP image OK"
```

模型、commit、镜像 digest 的固定合同见 [`docs/MODEL_ASSETS.md`](docs/MODEL_ASSETS.md)。若 FP 镜像不存在，先按 FoundationPose 上游 Docker 指南构建，再把 `SIM_S3_FP_IMAGE` 改为你的实际标签；不要用空容器或另一模型静默代替。

## 3. 建 Python 环境并生成 AMBF launch

```bash
cd "$HOME/suturing-policy-sim2real"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/analysis.txt
```

完整 AMBF/RL 运行还需要与你的 CUDA 匹配的 PyTorch，然后安装：

```bash
# 先用 PyTorch 官方命令安装与你的 CUDA 匹配的 torch，再执行：
python -m pip install -r requirements/simulation.txt
```

ROS Python 包、`PyKDL` 和 `ambf_msgs` 应来自 ROS/AMBF workspace，不要从随机同名 PyPI 包替代。加载它们：

```bash
source /opt/ros/humble/setup.bash
source "$HOME/ambf_ros_ws/install/setup.bash"
```

将 SRC 的真实路径写入发布的 AMBF 模板：

```bash
python scripts/prepare_ambf_launch.py \
  --src-root "$SRC_ROOT" \
  --output-dir "$HOME/.cache/surgicai/launch"
```

采用推荐输出目录时，`configs/pipeline.env` 的默认 `SIM_S3_STEREO_LAUNCH` / `SIM_S4_STEREO_LAUNCH` 已指向生成文件；否则把脚本打印出的路径填回配置。

## 4. 按层跑通主代码

### 4.1 L0：只验证代码

不启动 AMBF、DA、FP 或机器人：

```bash
python scripts/run_simulation.py --stage code --profile smoke
```

成功结尾：

```text
PIPELINE_COMPLETE stage=code profile=smoke ...
```

这一步验证 Python 逻辑、测试和 synthetic hand-eye 软件合同，不验证 GPU 感知或真机精度。

### 4.2 L1：冻结感知 bank → AMBF Reach

这一步重新启动 AMBF 和控制器，但复用仓库中的冻结 S3 bank，不重新计算 DA/FP：

```bash
set -a; source configs/pipeline.env; set +a
source /opt/ros/humble/setup.bash
source "$HOME/ambf_ros_ws/install/setup.bash"

python scripts/doctor.py \
  --profile reach \
  --config configs/pipeline.env \
  --check-domain-clean

python scripts/run_simulation.py \
  --stage s4 \
  --profile smoke \
  --goal both \
  --controller d2
```

`smoke` 只跑 2 episode，用来查接线。通过后再把 `--profile smoke` 改为 `--profile formal`。

### 4.3 L2：RGB → DA → FP gate → Reach

这是完整仿真主代码。先检查全部依赖：

```bash
python scripts/doctor.py \
  --profile full \
  --config configs/pipeline.env \
  --check-domain-clean
```

只有看到 `DOCTOR_PASS` 才运行：

```bash
python scripts/run_simulation.py \
  --stage full \
  --profile smoke \
  --depth da \
  --goal both \
  --controller d2
```

通过冒烟后跑固定正式合同：

```bash
python scripts/run_simulation.py \
  --stage full \
  --profile formal \
  --depth da \
  --goal both \
  --controller d2
```

正式顺序是：40 次 reset/capture → DA → FP candidates → 首帧物理门 → frozen goal bank → AMBF 重启 → GT/FP 两组各 30 episode Reach。

## 5. 如何替换 GT、DA、FP 和 controller

不要改内部 Python 文件；只改主命令的开关：

| 参数 | 选择 | 含义 |
|---|---|---|
| `--depth` | `gt` / `da` | AMBF 特权真深度 / DA 预测深度 |
| `--goal` | `gt` / `fp` / `both` | GT 针位姿 / gated FP 针位姿 / 同 reset 配对 A/B |
| `--controller` | `d2` / `rl` | 分阶段 SE(3) servo / 发布的 RL checkpoint |
| `--profile` | `smoke` / `formal` | 2 条接线测试 / 固定 40+30 正式合同 |

常用隔离实验：

```bash
# 只测试 FP：GT depth + FP goal
python scripts/run_simulation.py --stage full --profile smoke \
  --depth gt --goal fp --controller d2

# 绕过 FP：DA 仍计算，但控制使用 GT goal
python scripts/run_simulation.py --stage full --profile smoke \
  --depth da --goal gt --controller d2

# 同一个冻结 bank 下比较 D2 与 RL（分别运行）
python scripts/run_simulation.py --stage s4 --profile smoke \
  --goal fp --controller d2
python scripts/run_simulation.py --stage s4 --profile smoke \
  --goal fp --controller rl
```

注意：`--goal gt` 绕过 FP，并不能证明 DA/FP 成功；`--stage s4` 复用 bank，也不能证明感知重新计算。只有 `--stage full --depth da --goal fp|both` 才执行完整感知路径。

## 6. 怎么判断成功、怎么定位失败

每次主命令最后会打印结果目录，例如：

```text
PIPELINE_COMPLETE ... root=/home/<user>/surgicai_runs/pipeline_smoke_<timestamp>
```

复制该路径执行：

```bash
python scripts/inspect_results.py \
  "$HOME/surgicai_runs/pipeline_smoke_<timestamp>"
```

通过标志是 `RESULT_AUDIT_PASS`。不要只看 AMBF 窗口，也不要只看进程退出码。

主入口按失败层返回稳定错误码：

| 错误 | 失败层 | 先看哪里 |
|---|---|---|
| `D9-E03-CONFIG` | 配置文件/路径 | `configs/pipeline.env` |
| `D9-E10-PREFLIGHT` | ROS、AMBF、模型、domain | 结果目录的 `doctor.json` |
| `D9-E60-S3` | capture、DA、FP 或物理门 | `sim_s3/*.log`、`fp_gate/result.json` |
| `D9-E90-S4` | goal bank、AMBF 或控制 | `sim_s4/*/status.txt`、`result.json` |

进一步排错见 [`docs/zh/参数与排错.md`](docs/zh/参数与排错.md)。任何失败都保留原始结果目录；不要删失败 episode 后重算成功率。

## 7. 真机入口与仿真的区别

真机不是把 AMBF topic 名替换一下就结束。当前 ROS 2 runtime 位于 [`ros2_ws/src/suturing_runtime`](ros2_ws/src/suturing_runtime/README.md)，统一消费：

```text
/suturing/ecm/left/image
/suturing/depth/metric
/suturing/needle/mask
/suturing/needle/pose_gated
/suturing/psm1/measured_cp
```

并输出诊断、TF goal 和受保护的单次 Reach。它默认只读。R2 已包含经过 P5a 验收的
DA backend，但默认关闭，必须提供冻结 checkpoint 与 Depth-Anything-V2 路径；人工
polygon/PNG 首帧 mask publisher 已包含，自动分割和 FoundationPose backend 仍是外部
输入。只有 R6 平放门与人工确认之后的针位姿才会发布到
`/suturing/needle/pose_gated`。这是刻意的失败隔离，不是“完整真机自动 pipeline 已完成”。

编译和只读冒烟：

```bash
source /opt/ros/humble/setup.bash
mkdir -p "$HOME/suturing_ros_ws/src"
ln -s "$HOME/suturing-policy-sim2real/ros2_ws/src/suturing_runtime" \
  "$HOME/suturing_ros_ws/src/suturing_runtime"
cd "$HOME/suturing_ros_ws"
colcon build --symlink-install --packages-select suturing_runtime
source install/setup.bash

ros2 launch suturing_runtime real_read_only.launch.py
```

先完成 topic 类型、相机内参、hand-eye、frame/单位、mask、freshness、低速和人工停止验证，才可进入 guarded motion。真机与仿真输入逐项差异见 [`docs/zh/真机与仿真的区别.md`](docs/zh/真机与仿真的区别.md)；逐文件、逐 callback 的白话代码讲义见 [`docs/zh/真机主代码逐段讲义.md`](docs/zh/真机主代码逐段讲义.md)。

## 8. 本地验证范围

发布入口已做过：

- Windows 纯逻辑单元测试；
- GitHub clean clone 的 L0 主入口；
- WSL Ubuntu 22.04 / ROS 2 Humble 的 `suturing_runtime` colcon build；
- mock RGB/depth/mask/needle pose/PSM pose → TF goal → `READY_READ_ONLY`；
- 只读模式确认没有 raw dVRK servo publisher，未授权 motion 被拒绝。

没有在发布重构时连接真实 dVRK，也没有在一台全新 GPU 主机上重新跑 40+30 的 L2 正式矩阵。因此仓库内的历史 40/40 gate、29/30 Reach 是参考证据，不应写成“任意新机器已逐条复现”。证据文件在 [`docs/evidence/`](docs/evidence/)。

## License

项目生成源码采用 MIT。AMBF、SRC、FoundationPose、Depth Anything、mesh 和外部 checkpoint 遵循各自许可证。
