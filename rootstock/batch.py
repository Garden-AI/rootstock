"""
Batch convergence: the planner and executor behind ``rootstock sync``.

The planner diffs desired state — the env sources registered in
``{root}/environments/``, optionally overlaid by a staging directory of
new/patched sources, plus each source's CHECKPOINTS table — against actual
state (the InstallState reader + manifest), and emits only the delta as work
items. The executor runs those items in three ordered phases — build,
download, verify — each a thread pool over subprocess-heavy operations (so
plain threads suffice), with keep-going failure semantics: a failed item
skips its dependents with the cause attached, and whatever didn't converge
is picked up by simply re-running sync. The install root is the checkpoint;
there is no other resume state.
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import operations as ops
from .environment import is_custom_checkpoint, parse_checkpoints_dict
from .exceptions import RootstockError
from .install_state import read_install_state
from .manifest import compute_source_hash, is_verified
from .verify import DEFAULT_VERIFY_TIMEOUT

PHASES = ("build", "download", "verify")

Say = Callable[[str], None]


# -----------------------------------------------------------------------------
# Plan
# -----------------------------------------------------------------------------


@dataclass
class BuildItem:
    env_name: str
    source: str  # staged file path, or registered env name
    reason: str


@dataclass
class CheckpointItem:
    env_name: str
    checkpoint: str
    reason: str


@dataclass
class SyncPlan:
    """The delta between declared and actual state, as ordered work items."""

    builds: list[BuildItem] = field(default_factory=list)
    downloads: list[CheckpointItem] = field(default_factory=list)
    verifies: list[CheckpointItem] = field(default_factory=list)
    # Advisory findings that don't trigger work by themselves (e.g. rootstock
    # pin drift, which only --rebuild acts on).
    notes: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.builds or self.downloads or self.verifies)

    def to_dict(self) -> dict:
        return asdict(self)


def _desired_sources(
    state,
    source_dir: Path | None,
    envs: list[str] | None,
) -> dict[str, tuple[str, Path]]:
    """The env set sync converges toward: ``name -> (install_source, parse_path)``.

    ``install_source`` is what a build item hands to install_environment (a
    staged file path, or the registered name for name-mode rebuilds);
    ``parse_path`` is the source file whose CHECKPOINTS table will be live
    after this run — the staged file when one is provided, else the
    registered file.
    """
    desired: dict[str, tuple[str, Path]] = {}
    for name, path in state.sources:
        desired[name] = (name, path)

    if source_dir is not None:
        staged = sorted(source_dir.glob("*.py"))
        if not staged:
            raise ops.OperationError(f"No *.py environment files found in {source_dir}")
        for path in staged:
            desired[path.stem] = (str(path), path)

    if envs:
        unknown = sorted(set(envs) - set(desired) - set(state.envs))
        if unknown:
            raise ops.OperationError(
                f"Unknown environment(s): {', '.join(unknown)}. "
                f"Known: {', '.join(sorted(set(desired) | set(state.envs))) or '(none)'}"
            )
        desired = {name: spec for name, spec in desired.items() if name in envs}

    return desired


def plan_sync(
    root: Path,
    *,
    source_dir: Path | None = None,
    envs: list[str] | None = None,
    checkpoints: list[str] | None = None,
    rebuild: bool = False,
    phases: Sequence[str] = PHASES,
) -> SyncPlan:
    """Compute the work needed to converge ``root`` to its declared state.

    Triggers (see the design in issue #182):
      - build: env not built; staged source differs from the built
        ``env_source.py``; or ``rebuild``.
      - download: declared (non-``:custom``) checkpoint with no ``fetched_at``.
      - verify: newly fetched or rebuilt this run, or ``verified_at`` missing
        or older than ``built_at`` (the existing staleness rule).

    Rootstock-pin drift is reported as a note, never acted on implicitly —
    otherwise installing a newer CLI and syncing to add one checkpoint would
    surprise-rebuild every env on the cluster.
    """
    from . import __version__

    root = Path(root)
    state = read_install_state(root)
    plan = SyncPlan()

    desired = _desired_sources(state, source_dir, envs)

    # ---- build items -----------------------------------------------------
    rebuilt_envs: set[str] = set()
    for name in sorted(desired):
        install_source, parse_path = desired[name]
        built = state.envs.get(name)
        if built is None or not (built.path / "bin" / "python").exists():
            plan.builds.append(BuildItem(name, install_source, "not built"))
            rebuilt_envs.add(name)
        elif rebuild:
            plan.builds.append(BuildItem(name, install_source, "--rebuild"))
            rebuilt_envs.add(name)
        elif install_source != name:  # staged source provided
            if built.source_hash != compute_source_hash(parse_path):
                plan.builds.append(BuildItem(name, install_source, "source changed"))
                rebuilt_envs.add(name)

    # ---- pin-drift notes ---------------------------------------------------
    for name in sorted(desired):
        built = state.envs.get(name)
        if built is None or name in rebuilt_envs or built.record is None:
            continue
        pinned = built.record.dependencies.get("rootstock")
        if pinned and pinned != __version__:
            plan.notes.append(
                f"{name}: built with rootstock {pinned}, running CLI is "
                f"{__version__} (pass --rebuild to rebuild against it)"
            )

    # ---- checkpoint universe ----------------------------------------------
    declared: dict[tuple[str, str], None] = {}  # insertion-ordered set
    for name in sorted(desired):
        _, parse_path = desired[name]
        try:
            table = parse_checkpoints_dict(parse_path)
        except ValueError as exc:
            plan.notes.append(f"{name}: cannot parse CHECKPOINTS ({exc}); skipping its checkpoints")
            continue
        for ckpt_id in sorted(table):
            if is_custom_checkpoint(ckpt_id):
                continue  # nothing to download or verify; weights are the user's
            declared[(name, ckpt_id)] = None

    if checkpoints:
        known_ids = {ckpt for _, ckpt in declared}
        unknown = sorted(set(checkpoints) - known_ids)
        if unknown:
            raise ops.OperationError(
                f"Unknown checkpoint id(s): {', '.join(unknown)}. "
                f"Declared by the selected envs: {', '.join(sorted(known_ids)) or '(none)'}"
            )
        declared = {key: None for key in declared if key[1] in checkpoints}

    # ---- download / verify items -------------------------------------------
    manifest = state.manifest
    for env_name, ckpt_id in declared:
        record = manifest.environments.get(env_name) if manifest else None
        ckpt_record = record.checkpoints.get(ckpt_id) if record else None
        fetched = ckpt_record is not None and ckpt_record.fetched_at is not None

        needs_download = not fetched
        if needs_download:
            plan.downloads.append(CheckpointItem(env_name, ckpt_id, "not fetched"))

        if env_name in rebuilt_envs:
            verify_reason = "stale after rebuild"
        elif needs_download:
            verify_reason = "newly fetched"
        elif ckpt_record is None or ckpt_record.verified_at is None:
            verify_reason = "never verified"
        elif record is not None and not is_verified(record, ckpt_record):
            verify_reason = "stale (verified before last build)"
        else:
            continue
        plan.verifies.append(CheckpointItem(env_name, ckpt_id, verify_reason))

    # ---- phase selection ----------------------------------------------------
    if "build" not in phases:
        plan.builds = []
    if "download" not in phases:
        plan.downloads = []
    if "verify" not in phases:
        plan.verifies = []

    return plan


# -----------------------------------------------------------------------------
# Execution
# -----------------------------------------------------------------------------


@dataclass
class ItemResult:
    phase: str
    env_name: str
    checkpoint: str | None
    status: str  # "ok" | "failed" | "skipped"
    reason: str  # the plan reason (ok), error message (failed), or skip cause
    seconds: float | None = None

    @property
    def label(self) -> str:
        return self.env_name if self.checkpoint is None else f"{self.env_name}/{self.checkpoint}"


@dataclass
class SyncReport:
    results: list[ItemResult] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        out = {"ok": 0, "failed": 0, "skipped": 0}
        for result in self.results:
            out[result.status] += 1
        return out

    @property
    def failed(self) -> list[ItemResult]:
        return [r for r in self.results if r.status == "failed"]

    @property
    def skipped(self) -> list[ItemResult]:
        return [r for r in self.results if r.status == "skipped"]

    def to_dict(self) -> dict:
        return {
            "results": [asdict(r) for r in self.results],
            "counts": self.counts(),
        }


class _PhaseRunner:
    """Run one phase's items through a bounded thread pool.

    Parallel workers can't share a live stdout (uv stderr and per-item
    progress would interleave unreadably), so each item's progress lines are
    buffered and only surfaced — through the shared, locked ``say`` — when
    the item fails; successes get a one-line completion mark.
    """

    def __init__(self, phase: str, total: int, say: Say):
        self.phase = phase
        self.total = total
        self.say = say
        self._lock = threading.Lock()
        self._done = 0

    def _report(self, result: ItemResult, buffered: list[str]) -> None:
        with self._lock:
            self._done += 1
            mark = "ok" if result.status == "ok" else "FAILED"
            took = f", {result.seconds:.0f}s" if result.seconds is not None else ""
            self.say(f"[{self.phase} {self._done}/{self.total}] {result.label}: {mark}{took}")
            if result.status == "failed":
                for line in buffered:
                    self.say(f"    {line}")
                self.say(f"    error: {result.reason}")

    def run_item(self, item, work: Callable[[list[str]], None]) -> ItemResult:
        """Execute one item, buffering its progress; never raises."""
        checkpoint = getattr(item, "checkpoint", None)
        buffered: list[str] = []
        start = time.monotonic()
        try:
            work(buffered)
        except RootstockError as exc:
            result = ItemResult(self.phase, item.env_name, checkpoint, "failed", str(exc))
        except Exception as exc:  # keep-going: one item must never sink the batch
            result = ItemResult(self.phase, item.env_name, checkpoint, "failed", repr(exc))
        else:
            result = ItemResult(self.phase, item.env_name, checkpoint, "ok", item.reason)
        result.seconds = time.monotonic() - start
        self._report(result, buffered)
        return result


def _run_phase(
    phase: str,
    items: list,
    make_work: Callable,
    max_workers: int,
    fail_fast: bool,
    say: Say,
) -> list[ItemResult]:
    if not items:
        return []
    runner = _PhaseRunner(phase, len(items), say)
    results: list[ItemResult] = []
    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(items)))) as pool:
        futures = {pool.submit(runner.run_item, item, make_work(item)): item for item in items}
        pending = set(futures)
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                results.append(future.result())
            if fail_fast and any(r.status == "failed" for r in results):
                for future in pending:
                    if future.cancel():
                        item = futures[future]
                        results.append(
                            ItemResult(
                                phase,
                                item.env_name,
                                getattr(item, "checkpoint", None),
                                "skipped",
                                "--fail-fast",
                            )
                        )
                    else:
                        # Already running; let it finish and record honestly.
                        results.append(future.result())
                break
    return results


def execute_sync(
    root: Path,
    plan: SyncPlan,
    *,
    jobs: int = 4,
    verify_jobs: int = 1,
    verify_timeout: float = DEFAULT_VERIFY_TIMEOUT,
    device: str = "cuda",
    upgrade: bool = False,
    fail_fast: bool = False,
    push: bool = True,
    cache_root: Path | None = None,
    say: Say = print,
) -> SyncReport:
    """Execute a plan: build → download → verify, then one manifest refresh.

    Keep-going by default: a failed env build skips that env's checkpoint
    items (with the cause attached), a failed download skips that
    checkpoint's verify, and everything else proceeds. All operations run
    with ``push``/``refresh`` disabled; the full manifest refresh (and push,
    unless disabled) happens once at the end, when anything succeeded.
    """
    root = Path(root)
    report = SyncReport()

    # ---- phase 1: build ---------------------------------------------------
    def build_work(item: BuildItem):
        def work(buffered: list[str]) -> None:
            ops.install_environment(
                root,
                item.source,
                force=True,
                upgrade=upgrade,
                push=False,
                progress=buffered.append,
            )

        return work

    build_results = _run_phase("build", plan.builds, build_work, jobs, fail_fast, say)
    report.results.extend(build_results)
    failed_envs = {r.env_name for r in build_results if r.status != "ok"}
    aborted = fail_fast and any(r.status == "failed" for r in build_results)

    # ---- phase 2: download --------------------------------------------------
    downloads: list[CheckpointItem] = []
    for item in plan.downloads:
        if aborted:
            report.results.append(
                ItemResult("download", item.env_name, item.checkpoint, "skipped", "--fail-fast")
            )
        elif item.env_name in failed_envs:
            report.results.append(
                ItemResult(
                    "download",
                    item.env_name,
                    item.checkpoint,
                    "skipped",
                    f"env '{item.env_name}' did not build",
                )
            )
        else:
            downloads.append(item)

    def download_work(item: CheckpointItem):
        def work(buffered: list[str]) -> None:
            ops.fetch_checkpoint(
                root,
                item.checkpoint,
                cache_root=cache_root,
                refresh=False,
                progress=buffered.append,
            )

        return work

    download_results = _run_phase("download", downloads, download_work, jobs, fail_fast, say)
    report.results.extend(download_results)
    failed_downloads = {(r.env_name, r.checkpoint) for r in download_results if r.status != "ok"}
    aborted = aborted or (fail_fast and any(r.status == "failed" for r in download_results))

    # ---- phase 3: verify ------------------------------------------------------
    # Each verify loads a full model onto its device; the pool of device
    # names bounds how many are resident at once. `--verify-jobs N` with the
    # plain "cuda" device round-robins cuda:0..N-1 so each worker owns a GPU.
    device_pool: queue.Queue[str] = queue.Queue()
    if verify_jobs > 1 and device == "cuda":
        for i in range(verify_jobs):
            device_pool.put(f"cuda:{i}")
    else:
        for _ in range(max(1, verify_jobs)):
            device_pool.put(device)

    verifies: list[CheckpointItem] = []
    for item in plan.verifies:
        if aborted:
            report.results.append(
                ItemResult("verify", item.env_name, item.checkpoint, "skipped", "--fail-fast")
            )
        elif item.env_name in failed_envs:
            report.results.append(
                ItemResult(
                    "verify",
                    item.env_name,
                    item.checkpoint,
                    "skipped",
                    f"env '{item.env_name}' did not build",
                )
            )
        elif (item.env_name, item.checkpoint) in failed_downloads:
            report.results.append(
                ItemResult("verify", item.env_name, item.checkpoint, "skipped", "download failed")
            )
        else:
            verifies.append(item)

    def verify_work(item: CheckpointItem):
        def work(buffered: list[str]) -> None:
            dev = device_pool.get()
            try:
                ops.verify_fetched_checkpoint(
                    root,
                    item.checkpoint,
                    device=dev,
                    cache_root=cache_root,
                    refresh=False,
                    progress=buffered.append,
                    timeout=verify_timeout,
                )
            finally:
                device_pool.put(dev)

        return work

    verify_results = _run_phase("verify", verifies, verify_work, verify_jobs, fail_fast, say)
    report.results.extend(verify_results)

    # ---- final: one refresh + push -------------------------------------------
    if any(r.status == "ok" for r in report.results):
        ops.update_and_push_manifest(root, quiet=True, push=push)

    return report


def render_plan(plan: SyncPlan, say: Say = print) -> None:
    """Human-readable plan, terraform-style."""
    if plan.is_empty:
        say("Nothing to do — install matches its declared state.")
    for title, items in (
        ("build", plan.builds),
        ("download", plan.downloads),
        ("verify", plan.verifies),
    ):
        if not items:
            continue
        say(f"{title} ({len(items)}):")
        for item in items:
            if isinstance(item, BuildItem):
                label = item.env_name
            else:
                label = f"{item.env_name}/{item.checkpoint}"
            say(f"  {label:<40} {item.reason}")
    if plan.notes:
        say("notes:")
        for note in plan.notes:
            say(f"  - {note}")


def render_summary(report: SyncReport, say: Say = print) -> None:
    counts = report.counts()
    say("")
    say(f"Summary: {counts['ok']} ok, {counts['failed']} failed, {counts['skipped']} skipped")
    for result in report.failed:
        say(f"  FAILED  {result.phase}: {result.label} — {result.reason}")
    for result in report.skipped:
        say(f"  skipped {result.phase}: {result.label} — {result.reason}")
    if report.failed:
        say("Re-run the same sync after fixing; completed work is skipped automatically.")
