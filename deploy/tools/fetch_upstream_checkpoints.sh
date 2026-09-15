#!/usr/bin/env bash
# Fetch the released SurgicAI checkpoints this package knows the contract for.
#
# They are committed directly in surgical-robotics-ai/SurgicAI under
# RL/Evaluation_model/<task>/<algo>/final_model.zip, about 21-23 MB each, so
# they are not vendored here.  This pulls only the two TD3_HER_BC files the
# deployment uses, verifies their digests against contract.CHECKPOINT_CONTRACTS,
# and drops them in models/rl/upstream/.
#
#     bash tools/fetch_upstream_checkpoints.sh [destination]
set -euo pipefail

DEST="${1:-$(cd "$(dirname "$0")/../.." && pwd)/models/rl/upstream}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "sparse-cloning surgical-robotics-ai/SurgicAI into $TMP"
git clone --filter=blob:none --no-checkout --depth 1 \
    https://github.com/surgical-robotics-ai/SurgicAI.git "$TMP/SurgicAI"
cd "$TMP/SurgicAI"
git sparse-checkout set --no-cone \
    'RL/Evaluation_model/Approach/TD3_HER_BC/final_model.zip' \
    'RL/Evaluation_model/Place/TD3_HER_BC/final_model.zip' \
    'RL/Env_info'
git checkout

mkdir -p "$DEST"
cp RL/Evaluation_model/Approach/TD3_HER_BC/final_model.zip "$DEST/approach_td3_her_bc.zip"
cp RL/Evaluation_model/Place/TD3_HER_BC/final_model.zip    "$DEST/place_td3_her_bc.zip"

echo
echo "verifying against the digests in surgicai_rl_deploy/contract.py"
cd "$DEST"
python3 - "$DEST" <<'PY'
import sys, hashlib, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
root = pathlib.Path(sys.argv[1])
EXPECTED = {
    "approach_td3_her_bc.zip":
        "06fc7813f93feef5efa16d06e43c33f9750b4ef78d7ff921563205aaf4d2e71c",
    "place_td3_her_bc.zip":
        "4fc615f074158780744de28d189eddd7adc2bbdbbb8639f77582cbc2007eaff0",
}
bad = 0
for name, want in EXPECTED.items():
    h = hashlib.sha256((root / name).read_bytes()).hexdigest()
    ok = h == want
    bad += not ok
    print(f"  {'OK  ' if ok else 'FAIL'} {name}  {h[:16]}")
sys.exit(1 if bad else 0)
PY

echo
echo "checkpoints in $DEST"
echo "next:"
echo "  python3 tools/replay_demos.py --model $DEST/approach_td3_her_bc.zip --compare"
