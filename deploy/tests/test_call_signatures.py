"""Every keyword passed to a core constructor must exist on it.

A `TypeError: precheck() got an unexpected keyword argument` reached real
hardware. It could not be caught by the existing tests because the ROS node's
``main()`` needs rclpy and a robot, so nothing ever executed that call -- and
the offline tool, which is exercised end to end, happened to phrase the same
call differently and stayed correct.

Rather than try to run ``main()``, this reads the source. Every call to one of
the functions below, anywhere in the package or the tools, is checked against
that function's real signature. It is fast, needs nothing, and catches the
whole family: a renamed parameter, a keyword added to the wrong call site, a
argument that moved from one constructor to another.
"""

import ast
import inspect
from pathlib import Path

import pytest

from surgicai_rl_deploy import feasibility, loop, plan, sequence

ROOT = Path(__file__).resolve().parents[1]

#: name as it appears at the call site -> the callable it resolves to
WATCHED = {
    "precheck": feasibility.precheck,
    "build_plan": plan.build_plan,
    "LiftSpec": plan.LiftSpec,
    "TransportSpec": plan.TransportSpec,
    "GraspLiftPlan": plan.GraspLiftPlan,
    "SequenceConfig": sequence.SequenceConfig,
    "GraspLiftSequencer": sequence.GraspLiftSequencer,
    "LoopConfig": loop.LoopConfig,
    "SafetyLimits": loop.SafetyLimits,
    "ApproachLoop": loop.ApproachLoop,
}

SOURCES = sorted(
    [p for p in (ROOT / "surgicai_rl_deploy").glob("*.py")]
    + [p for p in (ROOT / "tools").glob("*.py")]
    + [ROOT / "run_pipeline.py", ROOT / "run_grasp_lift.py", ROOT / "run_approach.py"]
)


def call_sites(path: Path):
    """Yield (callable_name, lineno, [keywords]) for every watched call."""
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.id if isinstance(func, ast.Name)
            else func.attr if isinstance(func, ast.Attribute)
            else None
        )
        if name in WATCHED:
            # kw.arg is None for **kwargs unpacking, which we cannot check
            keywords = [kw.arg for kw in node.keywords if kw.arg is not None]
            yield name, node.lineno, keywords


def accepted(target) -> set:
    signature = inspect.signature(target)
    names = set()
    for param in signature.parameters.values():
        if param.kind is param.VAR_KEYWORD:
            return None  # accepts anything
        if param.kind is not param.VAR_POSITIONAL:
            names.add(param.name)
    return names


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_every_keyword_exists_on_the_thing_it_is_passed_to(path):
    problems = []
    for name, lineno, keywords in call_sites(path):
        allowed = accepted(WATCHED[name])
        if allowed is None:
            continue
        for keyword in keywords:
            if keyword not in allowed:
                problems.append(
                    f"{path.name}:{lineno}  {name}() has no parameter "
                    f"{keyword!r}  (it takes: {', '.join(sorted(allowed))})"
                )
    assert not problems, "\n" + "\n".join(problems)


def test_the_check_actually_looks_at_something():
    """A guard that silently matched nothing would be worse than none."""
    total = sum(len(list(call_sites(p))) for p in SOURCES)
    assert total > 20, f"only found {total} call sites; the AST walk is broken"


def test_the_check_would_have_caught_the_bug(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("precheck(plan, compensate_suture='apply')\n")
    found = [
        kw for _, _, kws in call_sites(bad) for kw in kws
        if kw not in accepted(feasibility.precheck)
    ]
    assert found == ["compensate_suture"]


# ======================================================================
# every args.X the code reads must be a flag the parser defines
# ======================================================================
def attribute_reads(path: Path, variable: str) -> set:
    tree = ast.parse(path.read_text(), filename=str(path))
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == variable
    }


def test_the_node_reads_only_arguments_it_defines(node_module):
    """An args.X with no matching --x is an AttributeError at run time.

    Same shape of bug as the one above, and the same reason the tests missed
    it: nothing executes the node's main().
    """
    defined = set(vars(node_module.parse_args(["--grasp-pos", "0", "0", "0"])))
    read = attribute_reads(ROOT / "surgicai_rl_deploy" / "grasp_lift_node.py", "args")
    missing = sorted(read - defined)
    assert not missing, f"the node reads args.{{{', '.join(missing)}}} but never defines them"


def test_the_offline_tool_reads_only_arguments_it_defines():
    import sys

    sys.path.insert(0, str(ROOT / "tools"))
    import offline_grasp_lift

    defined = set(vars(offline_grasp_lift.parse_args([
        "--start-pos", "0", "0", "0", "--start-quat", "0", "0", "0", "1",
        "--grasp-pos", "0", "0", "0",
    ])))
    read = attribute_reads(ROOT / "tools" / "offline_grasp_lift.py", "args")
    missing = sorted(read - defined)
    assert not missing, f"the tool reads args.{{{', '.join(missing)}}} but never defines them"
