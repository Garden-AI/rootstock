"""
Batch convergence: the planners and executors behind ``rootstock sync`` and
``rootstock prune``.

The sync planner diffs desired state — the env sources registered in
``{root}/environments/``, optionally overlaid by a staging directory of
new/patched sources, plus each source's CHECKPOINTS table — against actual
state (the InstallState reader + manifest), and emits only the delta as work
items. The executor runs those items in three ordered phases — build,
download, verify — each a thread pool over subprocess-heavy operations (so
plain threads suffice), with keep-going failure semantics: a failed item
skips its dependents with the cause attached, and whatever didn't converge
is picked up by simply re-running sync. The install root is the checkpoint;
there is no other resume state.

The prune planner takes the opposite half of the same diff — ``actual −
desired`` — plus an internal-GC tier that needs no declared state at all
(``.build`` leftovers, orphaned interpreters, stale lockfiles, ``.trash``,
the uv cache). Same keep-going executor idiom, same single end-of-run
manifest refresh + push. Deletion orders are chosen so a crash mid-run
strands only *unreferenced* state that a later run collects.
"""

from __future__ import annotations

import os
import queue
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import operations as ops
from .environment import is_custom_checkpoint, parse_checkpoints_dict, parse_clusters_list
from .exceptions import RootstockError
from .install_state import read_install_state
from .layout import resolve_cache_root
from .manifest import (
    CheckpointInfo,
    compute_source_hash,
    is_verified,
    load_manifest,
    manifest_lock,
    save_manifest,
)
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
    cluster: str | None = None,
) -> SyncPlan:
    """Compute the work needed to converge ``root`` to its declared state.

    Triggers (see the design in issue #182):
      - build: env not built; staged source differs from the built
        ``env_source.py``; or ``rebuild``.
      - download: declared (non-``:custom``) checkpoint with no ``fetched_at``.
      - verify: newly fetched or rebuilt this run, or ``verified_at`` missing
        or older than ``built_at`` (the existing staleness rule), judged
        against the current cluster's verification record (#208).

    ``cluster`` is the machine this sync runs on; resolved from the manifest
    or registry when omitted (an error on shared installs, where verification
    must not be attributed by guess). Envs whose CLUSTERS excludes it are
    still built and their checkpoints downloaded — those artifacts are shared
    — but their verifies are skipped with a note.

    Rootstock-pin drift is reported as a note, never acted on implicitly —
    otherwise installing a newer CLI and syncing to add one checkpoint would
    surprise-rebuild every env on the cluster.
    """
    from . import __version__

    root = Path(root)
    if cluster is None and "verify" in phases:
        cluster = ops.resolve_current_cluster(root)
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

    # ---- cluster-restricted envs (#208) --------------------------------------
    # Only a machine an env serves can verify it; builds and downloads are
    # shared artifacts and proceed regardless. This skip is env-level and
    # conservative; smoke-test resolves per id (checkpoint-first), which is
    # the eventual model here too — a universal env's overridden ids would
    # then verify via the variant instead of being planned twice.
    unservable: set[str] = set()
    if cluster is not None:
        for name in sorted(desired):
            _, parse_path = desired[name]
            try:
                restriction = parse_clusters_list(parse_path)
            except ValueError as exc:
                plan.notes.append(f"{name}: cannot parse CLUSTERS ({exc}); skipping its verifies")
                unservable.add(name)
                continue
            if restriction is not None and cluster not in restriction:
                plan.notes.append(
                    f"{name}: serves only {', '.join(restriction)} — skipping verify on {cluster}"
                )
                unservable.add(name)

    # ---- download / verify items -------------------------------------------
    manifest = state.manifest
    for env_name, ckpt_id in declared:
        record = manifest.environments.get(env_name) if manifest else None
        ckpt_record = record.checkpoints.get(ckpt_id) if record else None
        fetched = ckpt_record is not None and ckpt_record.fetched_at is not None

        needs_download = not fetched
        if needs_download:
            plan.downloads.append(CheckpointItem(env_name, ckpt_id, "not fetched"))

        if env_name in unservable:
            continue
        verified_at = (
            ckpt_record.verification(cluster).verified_at
            if ckpt_record is not None and cluster is not None
            else None
        )
        if env_name in rebuilt_envs:
            verify_reason = "stale after rebuild"
        elif needs_download:
            verify_reason = "newly fetched"
        elif verified_at is None or ckpt_record is None or cluster is None:
            verify_reason = "never verified"
        elif record is not None and not is_verified(record, ckpt_record, cluster):
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


class _LiveList(list):
    """A progress sink that records lines *and* surfaces them immediately.

    Only safe when the phase runs serially — live output from parallel
    workers would interleave unreadably, which is why the default path
    buffers instead.
    """

    def __init__(self, say: Say):
        super().__init__()
        self._say = say

    def append(self, line: str) -> None:
        super().append(line)
        self._say(f"    {line}")


class _PhaseRunner:
    """Run one phase's items through a bounded thread pool.

    Parallel workers can't share a live stdout (uv stderr and per-item
    progress would interleave unreadably), so each item's progress lines are
    buffered and only surfaced — through the shared, locked ``say`` — when
    the item fails; successes get a one-line completion mark.

    ``live`` inverts that for serial phases (prune): every progress line
    streams as it happens, so someone tailing a batch job's outfile sees the
    slow deletes moving — and, on a hang, what they hung on.
    """

    def __init__(self, phase: str, total: int, say: Say, live: bool = False):
        self.phase = phase
        self.total = total
        self.say = say
        self.live = live
        self._lock = threading.Lock()
        self._done = 0

    def _report(self, result: ItemResult, buffered: list[str]) -> None:
        with self._lock:
            self._done += 1
            mark = "ok" if result.status == "ok" else "FAILED"
            took = f", {result.seconds:.0f}s" if result.seconds is not None else ""
            self.say(f"[{self.phase} {self._done}/{self.total}] {result.label}: {mark}{took}")
            if result.status == "failed":
                if not self.live:  # live lines were already surfaced
                    for line in buffered:
                        self.say(f"    {line}")
                self.say(f"    error: {result.reason}")

    def run_item(self, item, work: Callable[[list[str]], None]) -> ItemResult:
        """Execute one item, buffering (or live-streaming) its progress;
        never raises."""
        checkpoint = getattr(item, "checkpoint", None)
        buffered: list[str] = _LiveList(self.say) if self.live else []
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
    live: bool = False,
) -> list[ItemResult]:
    if not items:
        return []
    runner = _PhaseRunner(phase, len(items), say, live=live)
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
    cluster: str | None = None,
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
                cluster=cluster,
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
                    cluster=cluster,
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


# -----------------------------------------------------------------------------
# Prune plan
# -----------------------------------------------------------------------------

DEFAULT_MIN_AGE_HOURS = 24.0


@dataclass
class PruneCheckpointItem:
    """Drop one checkpoint's manifest record, then its unshared weight files.

    ``files`` are cache-root-relative paths from the record's ``weight_files``
    minus anything a surviving checkpoint also recorded (HF blobs are shared
    within a repo dir) and minus files covered by a whole-repo-dir item.
    ``recorded=False`` means the checkpoint predates weight tracking: only the
    record is dropped, and its weights surface as unattributed cache instead.
    """

    env_name: str
    checkpoint: str
    reason: str
    recorded: bool = True
    files: list[str] = field(default_factory=list)
    reclaim_bytes: int = 0


@dataclass
class PruneWeightDirItem:
    """Remove a whole HF repo dir: every checkpoint attributed to it is
    being pruned, so unrecorded residue (unread snapshot files, refs, locks)
    goes with it."""

    path: str  # cache-root-relative, e.g. cache/huggingface/hub/models--x--y
    reason: str
    checkpoints: list[str] = field(default_factory=list)
    reclaim_bytes: int = 0

    @property
    def env_name(self) -> str:  # display label for ItemResult
        return Path(self.path).name


@dataclass
class PruneEnvItem:
    """Remove a built env (trash-rename, then rmtree) and, in SOURCE_DIR
    mode, unregister its source file + lockfile."""

    env_name: str
    reason: str
    reclaim_bytes: int = 0
    env_dir: bool = True  # False: registered source with no built env
    source_file: str | None = None  # environments/<name>.py to unregister


@dataclass
class GCItem:
    """One piece of internal garbage; ``path`` is absolute."""

    kind: str  # "build" | "trash" | "interpreter" | "lockfile" | "uv-cache" | "unattributed"
    path: str
    reason: str
    reclaim_bytes: int | None = None  # None: unknown until executed (uv-cache)

    @property
    def env_name(self) -> str:  # display label for ItemResult
        return f"{self.kind}:{Path(self.path).name}"


@dataclass
class PrunePlan:
    """``actual − desired``, as ordered delete items, plus internal GC."""

    checkpoints: list[PruneCheckpointItem] = field(default_factory=list)
    weight_dirs: list[PruneWeightDirItem] = field(default_factory=list)
    envs: list[PruneEnvItem] = field(default_factory=list)
    gc: list[GCItem] = field(default_factory=list)
    # Tier 2: cache/home contents no weight record attributes. Report-only
    # unless --deep graduated them into gc items.
    unattributed: list[dict] = field(default_factory=list)  # {"path", "bytes"}
    notes: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.checkpoints or self.weight_dirs or self.envs or self.gc)

    @property
    def reclaim_bytes(self) -> int:
        return (
            sum(i.reclaim_bytes for i in self.checkpoints)
            + sum(i.reclaim_bytes for i in self.weight_dirs)
            + sum(i.reclaim_bytes for i in self.envs)
            + sum(i.reclaim_bytes or 0 for i in self.gc)
        )

    def to_dict(self) -> dict:
        return asdict(self)


def _tree_stats(path: Path) -> tuple[int, float]:
    """(recursive apparent size, newest mtime in the tree); symlinks counted,
    never followed.

    The mtime is the newest of *any* entry, not the top dir's: a download
    writing deep inside an HF repo dir (``blobs/…``) never bumps the
    top-level mtime, and the age guard must still see it as active.
    """
    try:
        top = path.lstat()
    except OSError:
        return 0, 0.0
    if not path.is_dir() or path.is_symlink():
        return top.st_size, top.st_mtime
    total, newest = 0, top.st_mtime
    for dirpath, dirnames, filenames in os.walk(path):
        for name in filenames:
            try:
                st = (Path(dirpath) / name).lstat()
            except OSError:
                continue
            total += st.st_size
            newest = max(newest, st.st_mtime)
        for name in dirnames:
            try:
                newest = max(newest, (Path(dirpath) / name).lstat().st_mtime)
            except OSError:
                pass
    return total, newest


def _tree_bytes(path: Path) -> int:
    """Recursive apparent size; symlinks counted, never followed."""
    return _tree_stats(path)[0]


def format_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(value) < 1024:
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def _safe_weight_path(cache_base: Path, relpath: str) -> Path | None:
    """Resolve a recorded cache-relative path, refusing anything that could
    escape the cache root — a corrupted record must not become an rm outside
    the install."""
    rel = Path(relpath)
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        return None
    return cache_base / rel


def _hf_repo_dir(relpath: str) -> str | None:
    """The cache-relative HF repo dir containing ``relpath``, if any.

    HF hub keeps one ``models--org--name`` dir per repo, with blobs shared by
    every revision — the unit at which whole-dir removal is safe.
    """
    parts = Path(relpath).parts
    for i, part in enumerate(parts):
        if part.startswith(("models--", "datasets--")):
            return str(Path(*parts[: i + 1]))
    return None


def plan_prune(
    root: Path,
    *,
    source_dir: Path | None = None,
    envs: list[str] | None = None,
    checkpoints: list[str] | None = None,
    gc_only: bool = False,
    deep: bool = False,
    min_age_hours: float = DEFAULT_MIN_AGE_HOURS,
    cache_root: Path | None = None,
    progress: Say | None = None,
) -> PrunePlan:
    """Compute what pruning would remove from ``root``.

    Desired state mirrors sync's inputs: the sources registered in
    ``{root}/environments/`` — or, with ``source_dir``, exactly that
    directory, so ``sync dir/ && prune dir/`` converges a cluster to a
    declared set (including unregistering source files). Category-B internal
    garbage is collected in both modes; ``gc_only`` limits the plan to it.

    A declaring directory that exists but is *empty* affirmatively declares
    zero environments (everything installed is undeclared); a registry dir
    that doesn't exist declares nothing, and the plan degrades to the
    internal-GC tier alone.

    ``min_age_hours`` guards every category-B item: pid-liveness is
    meaningless on a shared filesystem (the build owning ``.build/<env>.<pid>``
    may be on another node), so anything mtime-newer than the guard is left
    alone and noted.

    ``progress`` receives a line before each potentially slow step — the
    per-item byte counts come from full tree walks, which on a loaded
    shared filesystem can each take minutes with nothing else to show.
    """
    root = Path(root)
    cache_base = Path(cache_root) if cache_root is not None else resolve_cache_root(root)

    def emit(line: str) -> None:
        if progress is not None:
            progress(line)

    emit(f"reading install state under {root}")
    state = read_install_state(root)
    manifest = state.manifest
    plan = PrunePlan()
    cutoff = time.time() - min_age_hours * 3600.0

    def young(path: Path) -> bool:
        try:
            return path.lstat().st_mtime > cutoff
        except OSError:
            return True  # can't stat it -> don't touch it

    registered = dict(state.sources)

    # ---- desired sources ----------------------------------------------------
    # Presence vs. absence of the declaring directory carries meaning. A
    # directory that exists but declares nothing is an affirmative "zero
    # environments" — the declarative retirement move ("delete the source,
    # prune") taken to its limit. An *absent* registry is no declaration at
    # all: there is nothing to converge to, so only the internal-GC tier
    # (which needs no declared state) runs. A nonexistent SOURCE_DIR stays a
    # usage error in the adapter — that's a typo, not a declaration.
    declarative = not gc_only
    if source_dir is not None:
        desired: dict[str, Path] = {path.stem: path for path in sorted(source_dir.glob("*.py"))}
    elif (root / "environments").is_dir():
        desired = registered
    else:
        desired = {}
        if declarative and (state.envs or (manifest is not None and bool(manifest.environments))):
            plan.notes.append(
                f"{root / 'environments'} does not exist — no declared state to "
                f"converge to; collecting internal garbage only (an empty "
                f"environments/ dir, by contrast, declares zero environments)"
            )
        declarative = False

    # ---- scope validation (mirrors sync) -------------------------------------
    known_envs = set(state.envs) | set(registered) | set(desired)
    if manifest is not None:
        known_envs |= set(manifest.environments)
    if envs:
        unknown = sorted(set(envs) - known_envs)
        if unknown:
            raise ops.OperationError(
                f"Unknown environment(s): {', '.join(unknown)}. "
                f"Known: {', '.join(sorted(known_envs)) or '(none)'}"
            )
    recorded_ids = {
        ckpt_id
        for record in (manifest.environments.values() if manifest else [])
        for ckpt_id in record.checkpoints
    }
    if checkpoints:
        unknown = sorted(set(checkpoints) - recorded_ids)
        if unknown:
            raise ops.OperationError(
                f"Unknown checkpoint id(s): {', '.join(unknown)}. "
                f"Recorded in the manifest: {', '.join(sorted(recorded_ids)) or '(none)'}"
            )

    # ---- env items ------------------------------------------------------------
    # --checkpoint alone narrows the run to checkpoint pruning: batch usage
    # passes --yes, so "prune this one checkpoint id" must not ride along
    # with env deletion nobody asked for. --env restores env pruning scoped
    # to the named envs.
    env_tier = declarative and (bool(envs) or not checkpoints)
    pruned_env_names: set[str] = set()
    if env_tier and not desired and (state.envs or registered):
        # Reachable only when the declaring dir exists (an absent registry
        # already degraded to gc-only above): a deliberate zero-environment
        # declaration. The plan-confirm gate still stands, but say it plainly.
        where = source_dir if source_dir is not None else root / "environments"
        plan.notes.append(
            f"{where} declares no environments — reading it as a deliberate "
            f"zero-environment state; everything installed will be removed"
        )
    if env_tier:
        for name in sorted(set(state.envs) | set(registered)):
            if name in desired or (envs and name not in envs):
                continue
            built = state.envs.get(name)
            reason = f"not in {source_dir}" if source_dir is not None else "no registered source"
            if built is not None:
                emit(f"sizing envs/{name}")
            # In default mode a cruft env by definition has no registered
            # source; only SOURCE_DIR mode unregisters source files.
            unregister = source_dir is not None and name in registered
            plan.envs.append(
                PruneEnvItem(
                    env_name=name,
                    reason=reason,
                    reclaim_bytes=_tree_bytes(built.path) if built else 0,
                    env_dir=built is not None,
                    source_file=str(registered[name]) if unregister else None,
                )
            )
            pruned_env_names.add(name)

    # ---- checkpoint items -----------------------------------------------------
    # Declared ids per desired env. A desired env whose CHECKPOINTS can't be
    # parsed keeps everything: sync's planner skips *adding* work on a parse
    # error, so prune must skip *removing* — the conservative direction flips.
    declared_by_env: dict[str, set[str] | None] = {}
    if declarative:
        for name in sorted(desired):
            try:
                declared_by_env[name] = set(parse_checkpoints_dict(desired[name]))
            except ValueError as exc:
                declared_by_env[name] = None
                plan.notes.append(
                    f"{name}: cannot parse CHECKPOINTS ({exc}); keeping all of its "
                    f"checkpoint records"
                )

    pruned: list[tuple[str, str, CheckpointInfo, str]] = []  # (env, ckpt, record, reason)
    if declarative and manifest is not None:
        for env_name in sorted(manifest.environments):
            if envs and env_name not in envs:
                continue
            record = manifest.environments[env_name]
            declared = declared_by_env.get(env_name)
            for ckpt_id in sorted(record.checkpoints):
                if checkpoints and ckpt_id not in checkpoints:
                    continue
                if is_custom_checkpoint(ckpt_id):
                    continue  # never fetched, no weights of ours to release
                info = record.checkpoints[ckpt_id]
                if env_name in pruned_env_names:
                    reason = "env pruned"
                elif env_name not in desired:
                    # Manifest-only record (env gone from disk, not registered):
                    # the refresh drops the env record; release the weights first.
                    reason = "env not installed"
                elif declared is None:
                    continue
                elif ckpt_id not in declared:
                    reason = "not declared"
                elif info.fetched_at is not None and _weights_all_missing(info, cache_base):
                    reason = "weights missing on disk"
                else:
                    continue
                pruned.append((env_name, ckpt_id, info, reason))

    if pruned:
        assert manifest is not None  # pruned items only come from manifest records
        pruned_keys = {(env, ckpt) for env, ckpt, _, _ in pruned}
        # Refcount every recorded file across the whole manifest: a file is
        # deletable only when no *surviving* checkpoint also recorded it.
        surviving_files: set[str] = set()
        repo_attribution: dict[str, set[tuple[str, str]]] = {}
        for env_name, record in manifest.environments.items():
            for ckpt_id, info in record.checkpoints.items():
                for entry in info.weight_files or []:
                    relpath = entry.get("path")
                    if not relpath:
                        continue
                    if (env_name, ckpt_id) not in pruned_keys:
                        surviving_files.add(relpath)
                    repo = _hf_repo_dir(relpath)
                    if repo is not None:
                        repo_attribution.setdefault(repo, set()).add((env_name, ckpt_id))

        # Whole-dir release trusts the attribution map to be complete — but a
        # *fetched* surviving checkpoint with no weight record (pre-tracking
        # install, capture never ran) could be using any repo dir without
        # appearing in it. One such survivor anywhere disables dir-level
        # release; per-file refcounted deletion still runs, and one
        # smoke-test run backfills the records. Custom checkpoints never
        # have records or shipped weights, and unfetched ones have no bytes
        # on disk — neither blinds the map.
        unrecorded_survivors = sorted(
            ckpt_id
            for env_name, record in manifest.environments.items()
            for ckpt_id, info in record.checkpoints.items()
            if (env_name, ckpt_id) not in pruned_keys
            and not is_custom_checkpoint(ckpt_id)
            and info.fetched_at is not None
            and not info.weight_files
        )
        if unrecorded_survivors:
            released_repos: set[str] = set()
            if repo_attribution:
                plan.notes.append(
                    "keeping HF repo dirs whole (per-file deletion only): surviving "
                    "checkpoint(s) have no weight records and could share them: "
                    f"{', '.join(unrecorded_survivors)}. One smoke-test run backfills "
                    "the records."
                )
        else:
            released_repos = {
                repo for repo, keys in repo_attribution.items() if keys <= pruned_keys
            }

        for env_name, ckpt_id, info, reason in pruned:
            weight_files = info.weight_files or []
            files: list[str] = []
            reclaim = 0
            for entry in weight_files:
                relpath = entry.get("path")
                if not relpath or relpath in surviving_files:
                    continue
                if _hf_repo_dir(relpath) in released_repos:
                    continue  # the whole-repo-dir item covers it
                target = _safe_weight_path(cache_base, relpath)
                if target is None:
                    plan.notes.append(f"{ckpt_id}: refusing suspicious recorded path {relpath!r}")
                    continue
                try:
                    reclaim += target.lstat().st_size
                except OSError:
                    continue  # already gone; nothing to delete or count
                files.append(relpath)
            plan.checkpoints.append(
                PruneCheckpointItem(
                    env_name=env_name,
                    checkpoint=ckpt_id,
                    reason=reason,
                    recorded=bool(weight_files),
                    files=files,
                    reclaim_bytes=reclaim,
                )
            )

        for repo in sorted(released_repos):
            repo_path = _safe_weight_path(cache_base, repo)
            if repo_path is None or not repo_path.exists():
                continue
            emit(f"sizing {repo}")
            plan.weight_dirs.append(
                PruneWeightDirItem(
                    path=repo,
                    reason="every checkpoint attributed to it is pruned",
                    checkpoints=sorted({ckpt for _, ckpt in repo_attribution[repo]}),
                    reclaim_bytes=_tree_bytes(repo_path),
                )
            )

    # ---- category B: internal GC (both modes, age-guarded) ---------------------
    age_skipped: dict[str, int] = {}

    def collect(kind: str, entries: list[Path], reason_for: Callable[[Path], str]) -> None:
        for entry in entries:
            if young(entry):
                age_skipped[kind] = age_skipped.get(kind, 0) + 1
                continue
            emit(f"sizing {entry}")
            plan.gc.append(GCItem(kind, str(entry), reason_for(entry), _tree_bytes(entry)))

    build_root = root / ".build"
    if build_root.is_dir():
        collect("build", sorted(build_root.iterdir()), lambda _: "leftover from a crashed build")

    trash_root = root / ".trash"
    if trash_root.is_dir():
        collect("trash", sorted(trash_root.iterdir()), lambda _: "leftover from a crashed prune")

    python_root = root / ".python"
    if python_root.is_dir():
        # An interpreter is live iff some built env's bin/python resolves into
        # it — including envs being pruned this run, so a failed env deletion
        # never leaves a broken venv; a newly orphaned interpreter is simply
        # collected by the next run.
        live: set[str] = set()
        for env_state in state.envs.values():
            try:
                real = (env_state.path / "bin" / "python").resolve()
                rel = real.relative_to(python_root.resolve())
            except (OSError, ValueError):
                continue
            if rel.parts:
                live.add(rel.parts[0])
        orphans = [entry for entry in sorted(python_root.iterdir()) if entry.name not in live]
        collect(
            "interpreter",
            orphans,
            lambda entry: (
                "stranded staging copy"
                if ".installing." in entry.name
                else "no built env resolves into it"
            ),
        )

    env_src_dir = root / "environments"
    if env_src_dir.is_dir():
        orphan_locks = [
            lock
            for lock in sorted(env_src_dir.glob("*.py.lock"))
            if not lock.with_suffix("").exists()
        ]
        collect("lockfile", orphan_locks, lambda _: "no matching environment source")

    for kind, count in sorted(age_skipped.items()):
        plan.notes.append(
            f"{kind}: left {count} entr{'y' if count == 1 else 'ies'} younger than "
            f"--min-age ({min_age_hours:g}h) alone"
        )

    uv_cache = root / ".uv-cache"
    if uv_cache.is_dir():
        plan.gc.append(GCItem("uv-cache", str(uv_cache), "delegate to `uv cache prune`", None))

    # ---- tier 2: unattributed cache/home contents ------------------------------
    # The records capture the read working set, not the download set, so
    # "delete everything unrecorded" would over-delete. Report it instead;
    # --deep graduates it to deletion for maintainers who trust the records.
    # Gated on `declarative`: with no registry dir at all, the root doesn't
    # look like a healthy install, and --deep especially shouldn't act on it.
    if declarative:
        attributed_files: set[str] = set()
        for record in manifest.environments.values() if manifest else []:
            for info in record.checkpoints.values():
                for entry in info.weight_files or []:
                    if entry.get("path"):
                        attributed_files.add(entry["path"])

        def is_attributed(rel: str) -> bool:
            prefix = rel + "/"
            return any(f == rel or f.startswith(prefix) for f in attributed_files)

        candidates: list[Path] = []
        cache_dir = cache_base / "cache"
        if cache_dir.is_dir():
            for child in sorted(cache_dir.iterdir()):
                if child.name == "huggingface":
                    hub = child / "hub"
                    if hub.is_dir():
                        candidates.extend(sorted(p for p in hub.iterdir() if p.is_dir()))
                else:
                    candidates.append(child)
        home_dir = cache_base / "home"
        if home_dir.is_dir():
            for child in sorted(home_dir.iterdir()):
                if child.name == ".cache" and child.is_dir():
                    candidates.extend(sorted(child.iterdir()))
                else:
                    candidates.append(child)

        if candidates:
            emit("scanning cache/home for unattributed contents")
        released = {repo for repo in (i.path for i in plan.weight_dirs)}
        deep_young = 0
        for candidate in candidates:
            rel = str(candidate.relative_to(cache_base))
            if rel in released or is_attributed(rel):
                continue
            emit(f"sizing {rel}")
            size, newest = _tree_stats(candidate)
            # The age guard applies to --deep with extra force: a checkpoint
            # another node is downloading *right now* is unattributed by
            # definition (records are written only after setup succeeds).
            # Judge by the newest mtime anywhere in the tree — the write
            # lands deep inside the dir, not on the candidate itself.
            if deep and newest <= cutoff:
                plan.gc.append(
                    GCItem("unattributed", str(candidate), "no weight record claims it", size)
                )
            else:
                if deep:
                    deep_young += 1
                plan.unattributed.append({"path": str(candidate), "bytes": size})
        if deep_young:
            plan.notes.append(
                f"--deep: left {deep_young} unattributed "
                f"entr{'y' if deep_young == 1 else 'ies'} younger than --min-age "
                f"({min_age_hours:g}h) alone — an in-flight download looks exactly "
                f"like this; reported above instead"
            )

    return plan


def _weights_all_missing(info, cache_base: Path) -> bool:
    """A record whose weights were hand-deleted: files recorded, none exist."""
    paths = [
        _safe_weight_path(cache_base, entry["path"])
        for entry in info.weight_files or []
        if entry.get("path")
    ]
    paths = [p for p in paths if p is not None]
    return bool(paths) and not any(p.exists() for p in paths)


# -----------------------------------------------------------------------------
# Prune execution
# -----------------------------------------------------------------------------


def _prune_empty_parents(start: Path, cache_base: Path) -> None:
    """rmdir upward from ``start`` until a non-empty dir or a cache top."""
    stops = {cache_base, cache_base / "cache", cache_base / "home"}
    current = start
    while current not in stops and cache_base in current.parents:
        try:
            current.rmdir()
        except OSError:
            return
        current = current.parent


def execute_prune(
    root: Path,
    plan: PrunePlan,
    *,
    fail_fast: bool = False,
    push: bool = True,
    cache_root: Path | None = None,
    say: Say = print,
) -> SyncReport:
    """Execute a prune plan: checkpoints → weight dirs → envs → gc, then one
    manifest refresh + push.

    Deletion orders are crash-safe by construction:

    - Checkpoint records are dropped (one manifest transaction) *before* any
      weight file: a crash strands unreferenced files, which surface as
      unattributed cache — files-first would strand a ``fetched_at`` record
      pointing at nothing and make verify fail confusingly.
    - Envs go disk-first via trash-rename (``envs/<name>`` →
      ``.trash/<name>.<ts>``, then rmtree): readers see atomic disappearance,
      the manifest refresh already drops records for missing envs, and a
      crashed prune leaves ``.trash`` for the next run's GC.

    Serial by design (v1): nothing here is a six-hour job, and delete storms
    on a shared filesystem don't get faster with threads.
    """
    root = Path(root)
    cache_base = Path(cache_root) if cache_root is not None else resolve_cache_root(root)
    report = SyncReport()

    # ---- records first: one manifest transaction for every pruned checkpoint --
    record_error: str | None = None
    if plan.checkpoints:
        say(f"dropping {len(plan.checkpoints)} checkpoint record(s) from the manifest")
        try:
            with manifest_lock(root):
                manifest = load_manifest(root)
                if manifest is None:
                    # The plan's checkpoint items came from a manifest, so it
                    # vanishing means the world changed under us; deleting the
                    # files anyway would break records-drop-before-files.
                    raise ops.OperationError("manifest disappeared between plan and execute")
                for item in plan.checkpoints:
                    record = manifest.environments.get(item.env_name)
                    if record is not None:
                        record.checkpoints.pop(item.checkpoint, None)
                save_manifest(manifest, root)
        except Exception as exc:
            record_error = f"could not drop checkpoint records: {exc}"

    if record_error is not None:
        # Files must not outlive their records' removal, so the whole weight
        # tier is off; envs and gc are disk-first and safe to continue.
        for item in plan.checkpoints:
            report.results.append(
                ItemResult("checkpoint", item.env_name, item.checkpoint, "failed", record_error)
            )
        for dir_item in plan.weight_dirs:
            report.results.append(
                ItemResult(
                    "weights", dir_item.env_name, None, "skipped", "checkpoint records not dropped"
                )
            )

    aborted = fail_fast and record_error is not None

    # ---- phase: per-checkpoint weight files ------------------------------------
    # Every action is logged *before* it runs: on a shared filesystem a
    # single unlink or rmtree can stall for minutes, and someone tailing a
    # batch job's outfile should see what it stalled on.
    def checkpoint_work(item: PruneCheckpointItem):
        def work(buffered: list[str]) -> None:
            if not item.files:
                buffered.append("record dropped; no weight files to delete")
                return
            for relpath in item.files:
                target = _safe_weight_path(cache_base, relpath)
                if target is None:
                    continue
                try:
                    size = format_bytes(target.lstat().st_size)
                except OSError:
                    size = "already gone"
                buffered.append(f"rm {relpath} ({size})")
                try:
                    target.unlink()
                except FileNotFoundError:
                    pass
                _prune_empty_parents(target.parent, cache_base)

        return work

    if record_error is None and not aborted:
        results = _run_phase(
            "checkpoint", plan.checkpoints, checkpoint_work, 1, fail_fast, say, live=True
        )
        report.results.extend(results)
        aborted = fail_fast and any(r.status == "failed" for r in results)

        def dir_work(item: PruneWeightDirItem):
            def work(buffered: list[str]) -> None:
                target = _safe_weight_path(cache_base, item.path)
                if target is not None and target.exists():
                    buffered.append(f"rm -r {item.path} ({format_bytes(item.reclaim_bytes)})")
                    shutil.rmtree(target)
                    _prune_empty_parents(target.parent, cache_base)
                else:
                    buffered.append(f"{item.path} already gone")

            return work

        if aborted:
            for dir_item in plan.weight_dirs:
                report.results.append(
                    ItemResult("weights", dir_item.env_name, None, "skipped", "--fail-fast")
                )
        else:
            results = _run_phase(
                "weights", plan.weight_dirs, dir_work, 1, fail_fast, say, live=True
            )
            report.results.extend(results)
            aborted = fail_fast and any(r.status == "failed" for r in results)

    # ---- phase: envs -------------------------------------------------------------
    def env_work(item: PruneEnvItem):
        def work(buffered: list[str]) -> None:
            if item.env_dir:
                env_path = root / "envs" / item.env_name
                if env_path.exists():
                    trash = root / ".trash"
                    trash.mkdir(exist_ok=True)
                    doomed = trash / f"{item.env_name}.{int(time.time())}"
                    buffered.append(f"mv envs/{item.env_name} -> .trash/{doomed.name}")
                    env_path.rename(doomed)
                    buffered.append(
                        f"rm -r .trash/{doomed.name} ({format_bytes(item.reclaim_bytes)})"
                    )
                    shutil.rmtree(doomed)
            if item.source_file:
                source = Path(item.source_file)
                buffered.append(f"unregistering {source.name} (+ lockfile)")
                source.unlink(missing_ok=True)
                Path(f"{source}.lock").unlink(missing_ok=True)

        return work

    if aborted:
        for item in plan.envs:
            report.results.append(ItemResult("env", item.env_name, None, "skipped", "--fail-fast"))
    else:
        results = _run_phase("env", plan.envs, env_work, 1, fail_fast, say, live=True)
        report.results.extend(results)
        aborted = fail_fast and any(r.status == "failed" for r in results)

    # ---- phase: internal gc --------------------------------------------------------
    def gc_work(item: GCItem):
        def work(buffered: list[str]) -> None:
            path = Path(item.path)
            if item.kind == "uv-cache":
                if shutil.which("uv") is None:
                    raise ops.OperationError("uv not found in PATH")
                buffered.append("running `uv cache prune`")
                proc = subprocess.run(
                    ["uv", "cache", "prune"],
                    env={**os.environ, "UV_CACHE_DIR": str(path)},
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                if proc.returncode != 0:
                    raise ops.OperationError(
                        f"uv cache prune failed: {proc.stderr.strip() or proc.stdout.strip()}"
                    )
                for line in (proc.stderr + proc.stdout).splitlines():
                    if line.strip():
                        buffered.append(f"uv: {line.strip()}")
                return
            size = f" ({format_bytes(item.reclaim_bytes)})" if item.reclaim_bytes else ""
            try:
                shown = path.relative_to(root)
            except ValueError:
                shown = path
            buffered.append(f"rm -r {shown}{size}")
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)

        return work

    if aborted:
        for item in plan.gc:
            report.results.append(ItemResult("gc", item.env_name, None, "skipped", "--fail-fast"))
    else:
        report.results.extend(_run_phase("gc", plan.gc, gc_work, 1, fail_fast, say, live=True))

    # ---- final: one refresh + push -------------------------------------------------
    if any(r.status == "ok" for r in report.results):
        say(
            "refreshing manifest (runs `uv pip list` per env)"
            + ("; pushing to backend" if push else "; push disabled")
        )
        ops.update_and_push_manifest(root, quiet=True, push=push)

    return report


def render_prune_plan(
    plan: PrunePlan,
    say: Say = print,
    root: Path | None = None,
    cache_root: Path | None = None,
) -> None:
    """Human-readable plan with per-item reclaimed bytes, terraform-style.

    ``root``/``cache_root`` only shorten display paths; the plan's own paths
    stay absolute (gc) or cache-relative (weights) for the executor.
    """

    def display(path: str) -> str:
        for base in (root, cache_root):
            if base is not None:
                try:
                    return str(Path(path).relative_to(base))
                except ValueError:
                    continue
        return path

    if plan.is_empty:
        say("Nothing to prune — install matches its declared state.")

    if plan.checkpoints:
        say(f"checkpoint records ({len(plan.checkpoints)}):")
        for item in plan.checkpoints:
            label = f"{item.env_name}/{item.checkpoint}"
            if not item.recorded:
                detail = "no weight record — weights become unattributed"
            elif item.files:
                detail = f"{len(item.files)} file(s), {format_bytes(item.reclaim_bytes)}"
            else:
                detail = "no unshared weight files"
            say(f"  {label:<44} {item.reason:<28} {detail}")

    if plan.weight_dirs:
        say(f"weight dirs ({len(plan.weight_dirs)}):")
        for item in plan.weight_dirs:
            say(f"  {item.path:<60} {format_bytes(item.reclaim_bytes)}")

    if plan.envs:
        say(f"envs ({len(plan.envs)}):")
        for item in plan.envs:
            what = "" if item.env_dir else " (source file only)"
            say(
                f"  {item.env_name:<44} {item.reason + what:<40} {format_bytes(item.reclaim_bytes)}"
            )

    if plan.gc:
        say(f"gc ({len(plan.gc)}):")
        for item in plan.gc:
            size = format_bytes(item.reclaim_bytes) if item.reclaim_bytes is not None else ""
            say(f"  {display(item.path):<60} {item.reason:<36} {size}")

    if plan.unattributed:
        total = sum(entry["bytes"] for entry in plan.unattributed)
        say(
            f"unattributed cache ({format_bytes(total)}) — no weight record claims "
            f"these; review manually (--deep deletes them):"
        )
        for entry in plan.unattributed:
            say(f"  {display(entry['path']):<60} {format_bytes(entry['bytes'])}")

    if not plan.is_empty:
        say(f"reclaims {format_bytes(plan.reclaim_bytes)}")

    if plan.notes:
        say("notes:")
        for note in plan.notes:
            say(f"  - {note}")
