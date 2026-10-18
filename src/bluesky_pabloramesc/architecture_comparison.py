"""Schedule architecture-tree leaves as bounded, isolated training subprocesses."""

import fcntl
import json
import math
import os
import signal
import subprocess
from collections import deque
from contextlib import contextmanager
from numbers import Real
from pathlib import Path
from time import perf_counter, sleep

from .actions import ControlConfig
from .architecture_reporting import write_architecture_report
from .architecture_tree import default_tree, expand_tree, setting_arguments
from .comparison import comparison_command


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


@contextmanager
def _comparison_lock(directory):
    with (directory / ".lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"Another comparison is using {directory}") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def comparison_plan(args):
    """Validate the entire tree and shared protocol before creating any processes."""
    for name in ("jobs", "threads_per_job", "timesteps"):
        if type(getattr(args, name)) is not int or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not args.devices or any(
        not isinstance(device, str) or not device.strip() for device in args.devices
    ):
        raise ValueError("devices must contain at least one device name")
    if len(args.devices) > args.jobs:
        raise ValueError("Provide at least one job slot per listed device")
    if args.timeout is not None and (
        not math.isfinite(args.timeout) or args.timeout <= 0
    ):
        raise ValueError("timeout must be finite and positive")
    if args.retry_failed and not args.resume:
        raise ValueError("--retry-failed requires --resume")
    if args.history_lengths is not None and (
        args.temporal is None or set(args.temporal) == {"none"}
    ):
        raise ValueError("--history-lengths requires temporal methods other than none")
    if args.multi_agent and args.algorithm not in ("ppo", "sac"):
        raise ValueError("Multi-agent comparison supports PPO and SAC")
    for name in ("seeds", "eval_seeds"):
        values = getattr(args, name)
        if (
            not values
            or any(type(value) is not int or value < 0 for value in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError(f"{name} must contain unique nonnegative integers")
    control = ControlConfig(
        heading_action=args.heading_action,
        speed_action=args.speed_action,
        observation_encoding=args.observation_encoding,
        observation_profile=args.observation_profile,
        speed_min=args.speed_min,
        speed_max=args.speed_max,
        decision_interval=args.decision_interval,
    )
    if args.multi_agent and control.decision_interval != 10:
        raise ValueError("Decision interval tuning supports single-agent runs only")
    if args.tree is not None and (
        args.temporal is not None or args.history_lengths is not None
    ):
        raise ValueError(
            "Set temporal/history options inside --tree when using a custom tree"
        )
    manifest = args.output_dir / "comparison.json"
    if args.tree is not None:
        tree = json.loads(args.tree.read_text())
    elif (
        args.resume
        and manifest.is_file()
        and args.temporal is None
        and args.history_lengths is None
    ):
        tree = json.loads(manifest.read_text())["tree"]
    else:
        tree = default_tree(args.temporal or ("none",), args.history_lengths or (8,))
    variants = expand_tree(tree, args.algorithm, args.multi_agent)
    protocol = {
        "algorithm": args.algorithm,
        "multi_agent": args.multi_agent,
        "timesteps": args.timesteps,
        "seeds": args.seeds,
        "eval_seeds": args.eval_seeds,
        "control_config": control.to_dict(),
        "variants": [
            {"name": variant.name, "settings": variant.settings} for variant in variants
        ],
    }
    return tree, variants, protocol


def architecture_command(args, variant, seed, directory, device):
    """Use the existing train/evaluate workflow, with separate checkpoint paths."""
    return [
        *comparison_command(
            args.algorithm,
            args.multi_agent,
            args.heading_action,
            args.speed_action,
            seed,
            args.eval_seeds,
            directory,
            args.timesteps,
            args.speed_min,
            args.speed_max,
            device,
            observation_encoding=args.observation_encoding,
            observation_profile=args.observation_profile,
            decision_interval=args.decision_interval,
        ),
        *setting_arguments(variant.settings),
    ]


def _evaluation(record, variant, protocol):
    """Require a complete matched scenario/agent set and matching run settings."""
    paths = sorted((Path(record["directory"]) / "reports").glob("*/run.json"))
    if not paths:
        return None
    if len(paths) != 1:
        raise ValueError(f"Expected one run report in {record['directory']}")
    report = json.loads(paths[0].read_text())
    if type(report) is not dict:
        raise ValueError(f"Expected a report object in {paths[0]}")
    if report.get("status") != "completed":
        return None
    expected = {
        "seed": record["training_seed"],
        "requested_timesteps": protocol["timesteps"],
        "algorithm": protocol["algorithm"].upper(),
        "control_config": protocol["control_config"],
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Run configuration mismatch in {paths[0]}")
    architecture = report.get("architecture_config", {})
    if type(architecture) is not dict:
        raise ValueError(f"Expected architecture settings in {paths[0]}")
    if any(
        architecture.get(key, "none" if key == "temporal" else None) != value
        for key, value in variant.settings.items()
    ):
        raise ValueError(f"Architecture mismatch in {paths[0]}")
    if protocol["multi_agent"]:
        from .multi_agent import N_AGENTS

        slots = N_AGENTS
    else:
        slots = 1
    rows = report.get("evaluation_scenarios", [])
    expected_pairs = {
        (seed, agent) for seed in protocol["eval_seeds"] for agent in range(slots)
    }
    try:
        pairs = [(row["scenario_seed"], row["agent_index"]) for row in rows]
        valid = len(pairs) == len(expected_pairs) and set(pairs) == expected_pairs
        valid &= all(
            isinstance(row["return"], Real)
            and math.isfinite(row["return"])
            and all(
                not isinstance(value, Real) or math.isfinite(value)
                for value in row.values()
            )
            for row in rows
        )
    except (KeyError, TypeError) as error:
        raise ValueError(f"Malformed evaluation rows in {paths[0]}") from error
    if not valid:
        raise ValueError(f"Incomplete or nonfinite evaluation scenarios in {paths[0]}")
    for key in ("training_seconds", "actual_timesteps", "steps_per_second"):
        value = report.get(key)
        if not isinstance(value, Real) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid {key} in {paths[0]}")
        record[key] = value
    if report["actual_timesteps"] < protocol["timesteps"]:
        raise ValueError(f"Training budget was not completed in {paths[0]}")
    record["report_path"] = str(paths[0])
    return [
        {**row, "variant": variant.name, "training_seed": record["training_seed"]}
        for row in rows
    ]


def _prepare_runs(args, variants, protocol, previous):
    """Reuse valid results and preserve unsuccessful attempts before explicit retry."""
    records, episodes = [], []
    indexed = {
        (record["variant"], record["training_seed"]): record for record in previous
    }
    for variant in variants:
        for seed in args.seeds:
            old = indexed.get((variant.name, seed))
            if old is not None:
                if old["status"] == "running" and _process_alive(old.get("pid")):
                    raise ValueError(
                        f"Training process {old['pid']} is still running for {variant.name}"
                    )
                try:
                    evaluation = _evaluation(old, variant, protocol)
                except (ValueError, OSError) as error:
                    if old["status"] == "completed" and not args.retry_failed:
                        raise
                    evaluation = None
                    old["error"] = str(error)
                if evaluation is not None and old.get("returncode") in (None, 0):
                    old.update(status="completed", reused=True)
                    records.append(old)
                    episodes.extend(evaluation)
                    continue
                if old["status"] == "completed" and not args.retry_failed:
                    raise ValueError(
                        f"Completed run lost its evaluation: {old['directory']}"
                    )
                if old["status"] != "queued" and not args.retry_failed:
                    old.update(
                        status="failed",
                        error="Unfinished attempt preserved; use --resume --retry-failed to train a fresh attempt",
                    )
                    records.append(old)
                    continue
            attempt = (
                (old["attempt"] + 1)
                if old and old["status"] != "queued"
                else (old["attempt"] if old else 1)
            )
            directory = (
                args.output_dir
                / "runs"
                / Path(*variant.path)
                / f"seed_{seed}"
                / f"attempt_{attempt}"
            )
            if directory.exists():
                raise ValueError(f"Refusing to overwrite existing attempt: {directory}")
            records.append(
                {
                    "variant": variant.name,
                    "training_seed": seed,
                    "status": "queued",
                    "attempt": attempt,
                    "directory": str(directory),
                    "log_path": str(directory / "process.log"),
                    "settings": variant.settings,
                }
            )
    return records, episodes


def _process_alive(pid):
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _stop_process(process):
    """Allow checkpoint saving on SIGINT, then terminate a stuck process group."""
    for sig, timeout in ((signal.SIGINT, 10), (signal.SIGTERM, 3), (signal.SIGKILL, 3)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            continue


def _interrupt_handler(signum, frame):
    raise KeyboardInterrupt


def _schedule(args, variants, protocol, records, episodes):
    """A single lightweight event loop dynamically refills bounded device slots."""
    queue = deque(record for record in records if record["status"] == "queued")
    slots = deque(range(args.jobs))
    active = {}
    by_name = {variant.name: variant for variant in variants}
    interrupted = False
    environment = os.environ.copy()
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[key] = str(args.threads_per_job)
    environment["PYTHONUNBUFFERED"] = "1"
    state = args.output_dir / "runs.json"
    _write_json(state, records)
    try:
        while queue or active:
            while queue and slots:
                record, slot = queue.popleft(), slots.popleft()
                variant = by_name[record["variant"]]
                device = args.devices[slot % len(args.devices)]
                directory = Path(record["directory"])
                command = architecture_command(
                    args, variant, record["training_seed"], directory, device
                )
                directory.mkdir(parents=True)
                record.update(
                    status="running", command=command, device=device, reused=False
                )
                try:
                    with Path(record["log_path"]).open("w") as stream:
                        process = subprocess.Popen(
                            command,
                            stdout=stream,
                            stderr=subprocess.STDOUT,
                            env=environment,
                            start_new_session=True,
                        )
                except OSError as error:
                    record.update(status="failed", error=str(error))
                    slots.append(slot)
                else:
                    record["pid"] = process.pid
                    active[slot] = (process, record, perf_counter())
                    print(
                        f"Started {variant.name}, seed {record['training_seed']} on {device}; log: {record['log_path']}",
                        flush=True,
                    )
                _write_json(state, records)
            changed = False
            for slot, (process, record, started) in list(active.items()):
                timed_out = (
                    args.timeout is not None
                    and perf_counter() - started >= args.timeout
                )
                if process.poll() is None and not timed_out:
                    continue
                if timed_out and process.poll() is None:
                    _stop_process(process)
                    record["error"] = "Run exceeded its wall-clock timeout"
                record.update(
                    returncode=process.returncode, wall_seconds=perf_counter() - started
                )
                record["status"] = "failed"
                if process.returncode == 0 and not timed_out:
                    try:
                        evaluation = _evaluation(
                            record, by_name[record["variant"]], protocol
                        )
                        if evaluation is None:
                            raise ValueError(
                                "Training exited without a completed evaluation report"
                            )
                    except (ValueError, OSError) as error:
                        record["error"] = str(error)
                    else:
                        episodes.extend(evaluation)
                        record["status"] = "completed"
                else:
                    record.setdefault(
                        "error", f"Training exited with code {process.returncode}"
                    )
                print(
                    f"{record['status'].capitalize()}: {record['variant']}, seed {record['training_seed']}",
                    flush=True,
                )
                del active[slot]
                slots.append(slot)
                changed = True
            if changed:
                _write_json(state, records)
            if active:
                sleep(0.05)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        for process, record, started in active.values():
            _stop_process(process)
            record.update(
                status="interrupted",
                returncode=process.returncode,
                wall_seconds=perf_counter() - started,
            )
        _write_json(state, records)
    return interrupted


def run_architecture_comparison(args):
    """Run/reuse every leaf × training seed and always preserve partial reports."""
    tree, variants, protocol = comparison_plan(args)
    print(
        f"Architecture comparison: {len(variants)} variants × {len(args.seeds)} seeds = {len(variants) * len(args.seeds)} runs; at most {args.jobs} concurrent subprocesses",
        flush=True,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "tree": tree,
                    "protocol": protocol,
                    "commands": [
                        architecture_command(
                            args,
                            variant,
                            seed,
                            args.output_dir
                            / "runs"
                            / Path(*variant.path)
                            / f"seed_{seed}"
                            / "attempt_1",
                            args.devices[0],
                        )
                        for variant in variants
                        for seed in args.seeds
                    ],
                },
                indent=2,
            )
        )
        return 0
    if args.resume:
        if not (args.output_dir / "comparison.json").is_file():
            raise ValueError(f"No comparison manifest found in {args.output_dir}")
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
    with _comparison_lock(args.output_dir):
        manifest = args.output_dir / "comparison.json"
        if args.resume:
            previous = json.loads(manifest.read_text())
            if previous["protocol"] != protocol:
                raise ValueError(
                    "Cannot change the tree, seeds, evaluation scenarios, or training protocol when reusing results"
                )
        else:
            _write_json(manifest, {"tree": tree, "protocol": protocol})
        state = args.output_dir / "runs.json"
        previous_runs = json.loads(state.read_text()) if state.is_file() else []
        records, episodes = _prepare_runs(args, variants, protocol, previous_runs)
        _write_json(
            args.output_dir / "execution.json",
            {
                "jobs": args.jobs,
                "devices": args.devices,
                "threads_per_job": args.threads_per_job,
                "timeout": args.timeout,
            },
        )
        previous_handler = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, _interrupt_handler)
        interrupted = False
        try:
            interrupted = _schedule(args, variants, protocol, records, episodes)
        finally:
            signal.signal(signal.SIGTERM, previous_handler)
            write_architecture_report(
                args.output_dir, episodes, records, variants, args.seeds
            )
        print(f"Comparison report: {args.output_dir / 'report.md'}", flush=True)
        return (
            130
            if interrupted
            else int(any(record["status"] != "completed" for record in records))
        )
