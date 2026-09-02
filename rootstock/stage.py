"""Stage packed env images (and checkpoint weights) to node-local disk.

The structural fix for cold starts on network filesystems (#180): the spawn
path downloads each env as **one sequential read of a compressed image**
(see :mod:`rootstock.pack`) and extracts it to node-local disk, after which
imports, ``stat`` calls, and weight mmaps never touch the network again —
the two costs page-cache prewarming can't remove (metadata RPCs, and warmth
eviction under memory pressure) simply disappear.

Graceful degradation is load-bearing: any missing piece — no staging dir
configured, no packed image, image stale relative to the env build, missing
tools, insufficient disk, a lost lock race that times out, any unexpected
error — falls back to the existing prewarm path. A cluster where nobody
configures anything behaves exactly as before this module existed.

Where the staging base comes from (first declaration wins, then must
validate — exist, be writable, and live on a *different* filesystem than
the install root — or staging is disabled):

1. ``ROOTSTOCK_STAGE_DIR`` (user override / experiments / A-B testing),
2. ``stage_dir`` in ``{root}/layout.json`` (the install's own declaration;
   may contain env vars like ``$SLURM_TMPDIR``, expanded here on the node),
3. ``Cluster.stage_dir`` in the registry (legacy fallback).

``ROOTSTOCK_NO_STAGE=1`` force-disables, mirroring ``ROOTSTOCK_NO_PREWARM``.

Layout under the resolved base (per-user — staged trees are exec'd, so they
are never shared between users)::

    {base}/rootstock/{user}/
    ├── envs-by-hash/
    │   ├── {archive sha256}/            # one extracted image (content-addressed:
    │   │   ├── envs/<name>/             #   reuse across spawns/jobs is free, and a
    │   │   └── .python/<interp>/        #   rebuilt env lands in a new dir)
    │   ├── {sha256}.lock                # one extractor per node; others wait
    │   └── {sha256}.partial.<pid>       # in-flight extraction (swept when dead)
    └── cache-mirror/                    # weight overlay, shared across checkpoints

The weight mirror is an **overlay, not a copy**: each checkpoint's
manifest-recorded ``weight_files`` (#177) are materialized locally, and
everything else in the shared cache appears as a symlink back to it. The
fallthrough matters — the records only cover mmap-visible files, so small
side files (HF ``config.json`` and friends) must still resolve, just at
ordinary read speed. Worker caches (``HOME``/``XDG_CACHE_HOME``/``HF_HOME``)
are repointed at the mirror only when the overlay succeeds.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from . import __version__
from .pack import IMAGE_FORMAT, _pid_alive, pack_tools_missing
from .prewarm import _fmt_bytes

STAGE_DIR_ENV = "ROOTSTOCK_STAGE_DIR"
NO_STAGE_ENV = "ROOTSTOCK_NO_STAGE"

# A lockfile older than this is a crashed extractor's leftover (holds are
# minutes even on a congested night); a waiter that hasn't seen the winner
# finish by the same deadline gives up and falls back to prewarm.
_LOCK_STALE_SECONDS = 900.0
_WAIT_POLL_SECONDS = 1.0

# Extraction preflight headroom over the recorded uncompressed size (tar
# rounding, filesystem overhead, the in-flight compressed stream).
_SPACE_HEADROOM = 1.2

# Never evict a staged env younger than this: its job is plausibly still
# running from it. Older entries are fair game oldest-first.
_EVICT_MIN_AGE_SECONDS = 6 * 3600.0


@dataclass
class StagedSpawn:
    """What a successful staging pass hands back to ``spawn_in_env``."""

    env_dir: Path  # staged replacement for {root}/envs/<name>
    cache_base: Path | None  # weight-mirror base to point caches at, if staged


def _log(message: str) -> None:
    """Staging progress goes to the *client's* stderr (unlike the prewarm,
    which runs inside the worker): it is visible in job logs while the worker
    is still being spawned, and bracket lines before/after each phase keep a
    stall distinguishable from a hang."""
    print(f"[Rootstock] {message}", file=sys.stderr, flush=True)


def _same_filesystem(a: Path, b: Path) -> bool:
    return os.stat(a).st_dev == os.stat(b).st_dev


def resolve_stage_base(root: Path) -> Path | None:
    """Resolve and validate the node-local staging base for an install.

    Returns None — staging disabled — when nothing is declared or the first
    declaration found fails validation. Deliberately does not fall through
    to later declarations: a declared-but-broken dir is a configuration
    problem to surface (via debug-level fallback behavior), not to paper
    over with a stale registry entry.
    """
    if os.environ.get(NO_STAGE_ENV):
        return None

    raw = os.environ.get(STAGE_DIR_ENV)
    if not raw:
        from .layout import read_declared_stage_dir

        raw = read_declared_stage_dir(root)
    if not raw:
        from .clusters import get_cluster, get_cluster_for_root

        cluster_name = get_cluster_for_root(root)
        if cluster_name is not None:
            raw = get_cluster(cluster_name).stage_dir
    if not raw:
        return None

    expanded = os.path.expandvars(os.path.expanduser(raw))
    if "$" in expanded:
        # An env var in the declaration didn't expand here (e.g.
        # $SLURM_TMPDIR outside a job) — that node has no staging dir.
        return None
    base = Path(expanded)
    try:
        if not base.is_dir() or not os.access(base, os.W_OK | os.X_OK):
            return None
        # A "node-local" path that is actually the same filesystem as the
        # install would double every cost this module exists to remove.
        if _same_filesystem(base, Path(root)):
            return None
    except OSError:
        return None
    return base


def _read_manifest_env(root: Path, env_name: str) -> dict | None:
    """Raw ``manifest.json`` read of one env record — same contract as the
    prewarm-path lookup (#178): this runs on the spawn path as an
    optimization hint, so it must never print migration notes, take locks,
    or refuse a newer schema. Anything unreadable means "no record"."""
    try:
        with open(Path(root) / "manifest.json") as f:
            data = json.load(f)
        env = data.get("environments", {}).get(env_name)
    except (OSError, ValueError, AttributeError):
        return None
    return env if isinstance(env, dict) else None


def _current_image_record(root: Path, env_record: dict | None) -> dict | None:
    """The env record's image entry, iff it describes the current build and
    the archive file is present. Mirrors
    :func:`rootstock.manifest.image_is_current` on the raw dict."""
    if env_record is None:
        return None
    image = env_record.get("image")
    built_at = env_record.get("built_at")
    if not (isinstance(image, dict) and isinstance(built_at, str)):
        return None
    packed_at = image.get("packed_at")
    if not (isinstance(packed_at, str) and packed_at >= built_at):
        return None
    if not (
        isinstance(image.get("path"), str)
        and isinstance(image.get("sha256"), str)
        and image.get("format") == IMAGE_FORMAT
        and isinstance(image.get("uncompressed_bytes"), int)
    ):
        return None
    if not (Path(root) / image["path"]).is_file():
        return None
    return image


def _recorded_weight_entries(env_record: dict | None, checkpoint: str) -> list | None:
    """The checkpoint's ``weight_files`` list from a pre-read raw env record
    (same semantics as ``environment._recorded_weight_files``, which reads
    the manifest itself for the prewarm path)."""
    if not isinstance(env_record, dict):
        return None
    record = (env_record.get("checkpoints") or {}).get(checkpoint) or {}
    weight_files = record.get("weight_files")
    return weight_files if isinstance(weight_files, list) else None


def _user_stage_root(base: Path) -> Path:
    """``{base}/rootstock/{user}``, created 0700: staged trees are exec'd, so
    they must not be writable — or trustingly reusable — across users.

    The shared ``{base}/rootstock`` intermediate is made sticky-1777 (the
    /tmp recipe, same as the usage spool): whoever stages first must not
    lock everyone else's leaf-mkdir out via their umask. chmod is
    best-effort — on an already-existing dir owned by someone else it
    fails, and the mkdir below then either works or disables staging for
    this user with a visible log line (via the callers' guards).
    """
    shared = base / "rootstock"
    shared.mkdir(exist_ok=True)
    try:
        os.chmod(shared, 0o1777)
    except OSError:
        pass
    user_root = shared / getpass.getuser()
    user_root.mkdir(exist_ok=True)
    os.chmod(user_root, 0o700)
    return user_root


def _sweep_partials(envs_root: Path) -> None:
    """Remove extraction leftovers whose owning process is gone (SIGKILL
    hygiene — same "leak into node-local temp, auto-cleaned" posture as the
    spawn tmp dir)."""
    for partial in envs_root.glob("*.partial.*"):
        try:
            pid = int(partial.name.rsplit(".", 1)[-1])
        except ValueError:
            continue
        try:
            stale_by_age = time.time() - partial.stat().st_mtime > _LOCK_STALE_SECONDS
        except OSError:
            continue  # renamed/removed mid-scan by its extractor
        if not _pid_alive(pid) or stale_by_age:
            shutil.rmtree(partial, ignore_errors=True)


def _mark_in_use(staged_root: Path) -> None:
    """Record this process as a user of a staged env, in
    ``{staged}/.users/<pid>``. The recording process is the *client* (the
    calculator / server process), which lives as long as any worker it
    spawns — so pid-aliveness of these files is the eviction shield for
    long MD runs. Dead pidfiles are cleaned during eviction scans; nothing
    removes them on exit, and nothing needs to."""
    users = staged_root / ".users"
    try:
        users.mkdir(exist_ok=True)
        (users / str(os.getpid())).touch()
    except OSError:
        pass  # marking is best-effort; min-age still shields young dirs


def _in_use(staged_root: Path) -> bool:
    """Whether any recorded user of a staged env is still alive (pruning
    dead pidfiles as a side effect)."""
    live = False
    try:
        pidfiles = list((staged_root / ".users").iterdir())
    except OSError:
        return False
    for pidfile in pidfiles:
        if pidfile.name.isdigit() and _pid_alive(int(pidfile.name)):
            live = True
        else:
            pidfile.unlink(missing_ok=True)
    return live


def _evict_lru(envs_root: Path, keep: Path, bytes_needed: int) -> None:
    """Free space by removing the oldest staged envs — never ``keep``,
    never anything younger than the min age, and never a dir some live
    client process is registered against (mtime alone can't shield a
    multi-day MD run on a persistent /tmp). Best-effort: rechecks free
    space after each removal."""
    try:
        entries = sorted(
            (d for d in envs_root.iterdir() if d.is_dir() and d != keep),
            key=lambda d: d.stat().st_mtime,
        )
    except OSError:
        return
    now = time.time()
    for entry in entries:
        if shutil.disk_usage(envs_root).free >= bytes_needed:
            return
        try:
            if now - entry.stat().st_mtime < _EVICT_MIN_AGE_SECONDS:
                break  # sorted oldest-first: everything after is younger
        except OSError:
            continue
        if _in_use(entry):
            continue  # a live client is running workers out of it
        _log(f"Stage evicting {entry.name} (LRU) to make room")
        shutil.rmtree(entry, ignore_errors=True)


class _StageLock:
    """O_EXCL lockfile granting one extractor per archive per node. Losing
    the race is not an error — the loser waits for the winner's atomic
    rename instead of duplicating a multi-GB extraction (four contending
    prewarm streams are exactly the pathology staging replaces)."""

    def __init__(self, path: Path):
        self.path = path
        self.acquired = False

    def try_acquire(self) -> bool:
        for _ in range(2):  # second try after clearing a stale lock
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if self._is_stale():
                    self.path.unlink(missing_ok=True)
                    continue
                return False
            with os.fdopen(fd, "w") as f:
                f.write(str(os.getpid()))
            self.acquired = True
            return True
        return False

    def _is_stale(self) -> bool:
        try:
            content = self.path.read_text().strip()
            age = time.time() - self.path.stat().st_mtime
        except OSError:
            return False  # vanished — holder just released it
        if age > _LOCK_STALE_SECONDS:
            return True
        return content.isdigit() and not _pid_alive(int(content))

    def release(self) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)
            self.acquired = False


class _FixupError(RuntimeError):
    """A staged tree could not be pointed at its local interpreter.

    Deterministic for a given (archive, client version) — unlike transient
    extraction failures — so stage_env caches it per sha and stops paying
    the multi-GB extract-and-discard on every subsequent spawn.
    """


def _remap_into_stage(value: str, root: Path, staged_root: Path) -> str | None:
    """Rewrite an absolute path under the shared ``root`` to its staged
    equivalent; None when it doesn't point under the root.

    Tries the root as spelled and as resolved, then — because uv bakes the
    *install-time* spelling into symlink targets and pyvenv.cfg, which on
    multi-alias mounts (/eagle vs /lus/eagle at ALCF) matches neither —
    resolves the value itself and retries against the resolved root.
    """
    for prefix in {str(root), str(Path(root).resolve())}:
        if value == prefix or value.startswith(prefix.rstrip("/") + "/"):
            return str(staged_root) + value[len(prefix.rstrip("/")) :]
    if os.path.isabs(value):
        try:
            real = str(Path(value).resolve())
        except OSError:
            return None
        root_real = str(Path(root).resolve()).rstrip("/")
        if real == root_real or real.startswith(root_real + "/"):
            return str(staged_root) + real[len(root_real) :]
    return None


def _fixup_staged_env(tree: Path, final_root: Path, root: Path, env_name: str) -> None:
    """Point the extracted venv (still in the ``tree`` partial dir) at the
    interpreter it will have once the tree is renamed to ``final_root``.

    The archive reproduces root-relative layout, but two things inside a
    venv name the shared root absolutely and would quietly put the
    interpreter and stdlib back on the network filesystem:

    - the ``bin/python*`` symlinks (uv writes absolute targets), and
    - ``pyvenv.cfg``'s ``home =`` line, which is where the stdlib is
      resolved from at startup.

    Targets are written in ``final_root`` terms — dangling until the atomic
    rename, which is exactly the publication point waiters key off — and
    verified through ``tree``. Raises RuntimeError when ``bin/python`` can't
    be made local: a staged env that still executes from Lustre is worse
    than falling back.
    """
    env_dir = tree / "envs" / env_name
    for entry in (env_dir / "bin").iterdir():
        if not entry.is_symlink():
            continue
        target = os.readlink(entry)
        if not os.path.isabs(target):
            continue
        remapped = _remap_into_stage(target, root, final_root)
        if remapped is None:
            if entry.name.startswith("python"):
                raise _FixupError(f"staged {entry.name} links outside the install root ({target})")
            continue
        entry.unlink()
        os.symlink(remapped, entry)

    pyvenv = env_dir / "pyvenv.cfg"
    if pyvenv.is_file():
        lines = []
        for line in pyvenv.read_text().splitlines():
            key, sep, value = line.partition("=")
            remapped = _remap_into_stage(value.strip(), root, final_root) if sep else None
            lines.append(f"{key.rstrip()} {sep} {remapped}" if remapped else line)
        pyvenv.write_text("\n".join(lines) + "\n")

    python = env_dir / "bin" / "python"
    target = os.readlink(python) if python.is_symlink() else None
    if target is None or not (tree / Path(target).relative_to(final_root)).exists():
        raise _FixupError("staged bin/python does not resolve after fixup")


def stage_env(root: Path, env_name: str, base: Path, env_record: dict | None = None) -> Path | None:
    """Materialize the env's packed image under ``base``; return the staged
    root (containing ``envs/<name>`` and ``.python/``) or None to fall back.

    Content-addressed by archive sha256: a warm dir is reused with zero
    reads, and a rebuilt env (new archive, new sha) lands beside the old one,
    which ages out via LRU eviction. ``env_record`` is the pre-read raw
    manifest env record, for callers that already have it. Never raises —
    the CLI calls this bare, and even the preamble (mkdir on a shared /tmp,
    disk_usage, lock files) can fail on a hostile node.
    """
    try:
        return _stage_env(root, env_name, base, env_record)
    except Exception as exc:  # noqa: BLE001 - staging must never fail the caller
        try:
            _log(
                f"Stage skipped ({env_name}): {type(exc).__name__}: {exc}; falling back to prewarm"
            )
        except Exception:
            pass
        return None


def _stage_env(
    root: Path, env_name: str, base: Path, env_record: dict | None = None
) -> Path | None:
    if env_record is None:
        env_record = _read_manifest_env(root, env_name)
    image = _current_image_record(root, env_record)
    if image is None:
        return None
    missing_tools = pack_tools_missing()
    if missing_tools:
        _log(f"Stage skipped ({env_name}): {missing_tools} not on PATH; falling back to prewarm")
        return None

    envs_root = _user_stage_root(base) / "envs-by-hash"
    envs_root.mkdir(parents=True, exist_ok=True)
    final = envs_root / image["sha256"]
    marker = final / "envs" / env_name / "bin" / "python"

    if marker.exists():
        final.touch()  # LRU freshness
        _mark_in_use(final)
        _log(f"Stage reused (warm): {env_name} at {final}")
        return final

    # A recorded deterministic fixup failure for this archive + client
    # version: don't repeat a multi-GB extract-and-discard on every spawn.
    failed_note = envs_root / f"{image['sha256']}.failed"
    try:
        failed_version, failed_reason = failed_note.read_text().splitlines()[:2]
    except (OSError, ValueError):
        failed_version = None
        failed_reason = ""
    if failed_version == __version__:
        _log(
            f"Stage skipped ({env_name}): previously failed on this node "
            f"({failed_reason}); falling back to prewarm"
        )
        return None
    failed_note.unlink(missing_ok=True)  # other-version note: retry below

    _sweep_partials(envs_root)

    needed = int(image["uncompressed_bytes"] * _SPACE_HEADROOM)
    if shutil.disk_usage(envs_root).free < needed:
        _evict_lru(envs_root, keep=final, bytes_needed=needed)
        if shutil.disk_usage(envs_root).free < needed:
            _log(
                f"Stage skipped ({env_name}): needs {_fmt_bytes(needed)} free "
                f"at {envs_root}; falling back to prewarm"
            )
            return None

    lock = _StageLock(envs_root / f"{image['sha256']}.lock")
    if not lock.try_acquire():
        # Another spawn on this node is extracting the same archive; one
        # image stream then N local reuses beats N contending streams.
        _log(f"Staging of {env_name} in progress by another process; waiting")
        deadline = time.monotonic() + _LOCK_STALE_SECONDS
        while time.monotonic() < deadline:
            if marker.exists():
                _mark_in_use(final)
                _log(f"Stage reused (warm): {env_name} at {final}")
                return final
            if lock.try_acquire():
                break  # winner died; take over below
            time.sleep(_WAIT_POLL_SECONDS)
        if not lock.acquired:
            _log(f"Stage skipped ({env_name}): timed out waiting; falling back to prewarm")
            return None

    partial = envs_root / f"{image['sha256']}.partial.{os.getpid()}"
    image_path = Path(root) / image["path"]
    try:
        if marker.exists():  # completed while we raced for the lock
            _mark_in_use(final)
            return final
        _log(
            f"Staging {env_name} ({_fmt_bytes(image.get('compressed_bytes', 0))} "
            f"compressed) to {final}"
        )
        began = time.monotonic()
        partial.mkdir()
        zstd = subprocess.Popen(
            ["zstd", "-dc", str(image_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        tar = subprocess.Popen(
            ["tar", "-xf", "-", "-C", str(partial)],
            stdin=zstd.stdout,
            stderr=subprocess.PIPE,
        )
        assert zstd.stdout is not None
        zstd.stdout.close()
        tar_err = tar.communicate()[1]
        zstd_err = zstd.communicate()[1]
        if zstd.returncode != 0 or tar.returncode != 0:
            err = (zstd_err or tar_err).decode(errors="replace").strip()
            raise RuntimeError(f"extraction failed: {err}")

        _fixup_staged_env(partial, final, Path(root), env_name)

        try:
            partial.rename(final)
        except OSError:
            if not marker.exists():  # a real failure, not a lost race
                raise
        _mark_in_use(final)
        _log(f"Staged {env_name} in {time.monotonic() - began:.1f}s")
        return final
    except Exception as exc:  # noqa: BLE001 - staging must never fail the spawn
        if isinstance(exc, _FixupError):
            # Deterministic for this archive + client version; note it so
            # later spawns skip straight to prewarm instead of re-paying
            # the extraction. A client upgrade invalidates the note.
            try:
                failed_note.write_text(f"{__version__}\n{exc}\n")
            except OSError:
                pass
        _log(f"Stage skipped ({env_name}): {exc}; falling back to prewarm")
        shutil.rmtree(partial, ignore_errors=True)
        return None
    finally:
        lock.release()


# -----------------------------------------------------------------------------
# Weight overlay
# -----------------------------------------------------------------------------


def _hub_sibling_dirs(needed: list[PurePosixPath]) -> set[PurePosixPath]:
    """HuggingFace-hub special case: recorded weights are blob files, but the
    worker opens them through ``snapshots/<rev>/<file>`` relative symlinks
    (``../../blobs/<hash>``). If the repo's ``snapshots``/``refs`` dirs were
    left as whole-directory symlinks into the shared tree, those relative
    links would resolve to the *shared* blobs and the local copies would
    never be read — so the sibling dirs must be recreated in the mirror."""
    extras: set[PurePosixPath] = set()
    for rel in needed:
        parts = rel.parts
        if "blobs" in parts[:-1]:
            repo = PurePosixPath(*parts[: parts.index("blobs")])
            extras.update({repo / "snapshots", repo / "refs"})
    return extras


def _copy_file_atomic(src: Path, dest: Path, src_stat: os.stat_result | None = None) -> None:
    """Copy via tmp + rename. When ``src_stat`` is given, the source's mtime
    is preserved on the copy — that is the mirror's staleness signal: weight
    captures record only {path, size}, so a same-size in-place overwrite on
    the shared cache (a retrained ``:custom``-adjacent file, a torch-hub
    refresh) would otherwise serve stale bytes from a warm mirror forever."""
    tmp = dest.with_name(dest.name + f".copying.{os.getpid()}")
    try:
        shutil.copyfile(src, tmp)
        if src_stat is not None:
            os.utime(tmp, ns=(src_stat.st_atime_ns, src_stat.st_mtime_ns))
        tmp.rename(dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _mirror_current(mirror: Path, rel: str, src_stat: os.stat_result) -> bool:
    """Whether the mirror's copy of ``rel`` matches the shared source (size
    and preserved mtime). A fallthrough *symlink* is never current — its
    stat would trivially match the very shared file it points at, and
    treating it as a copy would leave a later-recorded weight file on the
    shared filesystem with the worker's prewarm switched off (worse than no
    staging). ``_copy_file_atomic``'s rename-over replaces the link."""
    dest = mirror / rel
    if os.path.islink(dest):
        return False
    try:
        st = os.stat(dest)
    except OSError:
        return False
    return st.st_size == src_stat.st_size and st.st_mtime_ns == src_stat.st_mtime_ns


def _weights_digest(sources: list[tuple[str, os.stat_result]]) -> str:
    """Identity of one checkpoint's overlay: the recorded paths plus each
    shared source's (size, mtime). Any source change — or a different
    record — produces a new digest and forces a re-overlay."""
    entries = sorted((rel, st.st_size, st.st_mtime_ns) for rel, st in sources)
    return hashlib.sha256(json.dumps(entries).encode()).hexdigest()


def _overlay_tree(shared: Path, mirror: Path, rel_dir: PurePosixPath) -> None:
    """Deep-copy one small subtree's *structure*: dirs recreated, symlink
    entries copied verbatim (their relative targets then resolve inside the
    mirror), small regular files copied. Used only for the hub sibling dirs,
    which hold pointers, not weights."""
    src_dir = shared / rel_dir
    dest_dir = mirror / rel_dir
    if dest_dir.is_symlink():
        dest_dir.unlink()
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        entries = list(src_dir.iterdir())
    except OSError:
        return
    for entry in entries:
        dest = dest_dir / entry.name
        if entry.is_symlink():
            if dest.is_symlink() or dest.exists():
                continue
            os.symlink(os.readlink(entry), dest)
        elif entry.is_dir():
            _overlay_tree(shared, mirror, rel_dir / entry.name)
        else:
            if not dest.exists():
                _copy_file_atomic(entry, dest)


def _overlay_recorded(
    shared_base: Path, mirror: Path, sources: list[tuple[str, os.stat_result]]
) -> int:
    """Build/refresh the weight overlay; returns bytes copied this pass.

    Ancestor directories of recorded files are materialized as real dirs;
    at each level, siblings not on any recorded path become symlinks back
    into the shared cache (the fallthrough for unrecorded side files). An
    existing real dir/file in the mirror is never downgraded to a symlink —
    it may be another checkpoint's materialized copy.
    """
    needed = {PurePosixPath(rel): st for rel, st in sources}

    materialize: set[PurePosixPath] = {PurePosixPath("cache"), PurePosixPath("home")}
    for rel in needed:
        materialize.update(p for p in rel.parents if p.parts)
    hub_siblings = _hub_sibling_dirs(list(needed))

    copied = 0
    for rel_dir in sorted(materialize, key=lambda p: len(p.parts)):
        src_dir = shared_base / rel_dir
        dest_dir = mirror / rel_dir
        if dest_dir.is_symlink():
            dest_dir.unlink()
        dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            entries = list(src_dir.iterdir())
        except OSError:
            continue  # shared side absent (e.g. bare home/): an empty local dir is right
        for entry in entries:
            rel_child = rel_dir / entry.name
            dest = mirror / rel_child
            if rel_child in materialize or rel_child in hub_siblings:
                continue  # handled by its own pass
            if rel_child in needed:
                src_stat = needed[rel_child]
                try:
                    if not _mirror_current(mirror, str(rel_child), src_stat):
                        _copy_file_atomic(entry, dest, src_stat)
                        copied += src_stat.st_size
                except OSError:
                    raise RuntimeError(f"could not copy recorded weight file {rel_child}")
            elif not (dest.is_symlink() or dest.exists()):
                os.symlink(entry, dest)

    for rel_dir in sorted(hub_siblings, key=lambda p: len(p.parts)):
        _overlay_tree(shared_base, mirror, rel_dir)
    return copied


def stage_weights(
    root: Path,
    cache_root: Path | None,
    env_name: str,
    checkpoint: str,
    base: Path,
    env_record: dict | None = None,
) -> Path | None:
    """Overlay the checkpoint's recorded weight files into the node-local
    cache mirror; return the mirror base or None to leave the worker's
    caches on the shared filesystem. Never raises (see stage_env).

    Only the manifest record tier is trusted here (unlike the prewarm's
    heuristic tier): redirecting a worker's caches at a mirror is only safe
    when we know exactly which files it will mmap. Every recorded file must
    exist in the shared cache — a purged file means the record is stale and
    the whole overlay is skipped, self-healing on the next add/verify pass.

    A per-checkpoint completion marker records the digest of the last
    finished overlay (recorded paths + shared sizes/mtimes). Matching it is
    the lock-free warm path: concurrent same-checkpoint spawns on one node —
    the committee-demo case — read the marker and return without ever
    touching the mirror lock or re-walking the shared cache.
    """
    try:
        return _stage_weights(root, cache_root, env_name, checkpoint, base, env_record)
    except Exception as exc:  # noqa: BLE001 - staging must never fail the caller
        try:
            _log(f"Weights not staged ({checkpoint}): {type(exc).__name__}: {exc}")
        except Exception:
            pass
        return None


def _stage_weights(
    root: Path,
    cache_root: Path | None,
    env_name: str,
    checkpoint: str,
    base: Path,
    env_record: dict | None = None,
) -> Path | None:
    if env_record is None:
        env_record = _read_manifest_env(root, env_name)
    recorded = _recorded_weight_entries(env_record, checkpoint)
    if not recorded:
        return None
    rels: list[str] = []
    for entry in recorded:
        if not (isinstance(entry, dict) and isinstance(entry.get("path"), str) and entry["path"]):
            return None
        rels.append(entry["path"])

    shared_base = Path(cache_root) if cache_root is not None else Path(root)
    sources: list[tuple[str, os.stat_result]] = []
    for rel in rels:
        try:
            st = os.stat(shared_base / rel)
        except OSError:
            return None  # purged / stale record — self-heals on the next verify
        sources.append((rel, st))

    user_root = _user_stage_root(base)
    mirror = user_root / "cache-mirror"
    marker = user_root / f"cache-mirror.{checkpoint}.ok"
    digest = _weights_digest(sources)

    try:
        if marker.read_text().strip() == digest:
            return mirror  # completed overlay, sources unchanged since
    except OSError:
        pass

    missing = sum(st.st_size for rel, st in sources if not _mirror_current(mirror, rel, st))
    if missing and shutil.disk_usage(user_root).free < missing * 1.1:
        _log(
            f"Weights not staged ({checkpoint}): needs {_fmt_bytes(missing * 1.1)} free at {mirror}"
        )
        return None

    # One overlay mutator per node: concurrent spawns touching one family's
    # dirs would race on the symlink/materialize transitions.
    lock = _StageLock(user_root / "cache-mirror.lock")
    deadline = time.monotonic() + _LOCK_STALE_SECONDS
    while not lock.try_acquire():
        if time.monotonic() > deadline:
            return None
        time.sleep(_WAIT_POLL_SECONDS)
    try:
        mirror.mkdir(parents=True, exist_ok=True)
        began = time.monotonic()
        copied = _overlay_recorded(shared_base, mirror, sources)
        # Marker only after the whole overlay (copies, fallthrough symlinks,
        # hub siblings) finished — a crash mid-overlay leaves no marker, so
        # the next spawn takes the lock and completes it.
        marker_tmp = marker.with_name(marker.name + f".{os.getpid()}")
        marker_tmp.write_text(digest)
        marker_tmp.rename(marker)
        if copied:
            _log(
                f"Staged weights for {checkpoint}: {_fmt_bytes(copied)} "
                f"in {time.monotonic() - began:.1f}s"
            )
        else:
            _log(f"Weights reused (warm) for {checkpoint}")
        return mirror
    except Exception as exc:  # noqa: BLE001 - overlay failure must not fail the spawn
        _log(f"Weights not staged ({checkpoint}): {exc}")
        return None
    finally:
        lock.release()


# -----------------------------------------------------------------------------
# The spawn seam
# -----------------------------------------------------------------------------


def stage_for_spawn(
    root: Path,
    env_name: str,
    payload: dict,
    cache_root: Path | None = None,
) -> StagedSpawn | None:
    """Best-effort staging pass for one worker spawn; never raises.

    The env stages for every worker spawn. Weights stage — and the worker's
    caches are repointed — only when the payload names a checkpoint and does
    *not* request weight capture: capture runs record files relative to the
    shared cache root, and add/verify passes must observe (and write) the
    shared cache, not a node-local mirror.
    """
    try:
        base = resolve_stage_base(Path(root))
        if base is None:
            return None
        env_record = _read_manifest_env(root, env_name)
        staged_root = stage_env(Path(root), env_name, base, env_record=env_record)
        if staged_root is None:
            return None

        cache_base: Path | None = None
        if payload.get("checkpoint") and "weights_capture" not in payload:
            cache_base = stage_weights(
                root, cache_root, env_name, payload["checkpoint"], base, env_record=env_record
            )
        return StagedSpawn(env_dir=staged_root / "envs" / env_name, cache_base=cache_base)
    except Exception as exc:  # noqa: BLE001 - staging must never fail the spawn
        try:
            _log(f"Stage skipped ({env_name}): {type(exc).__name__}: {exc}")
        except Exception:
            pass
        return None
