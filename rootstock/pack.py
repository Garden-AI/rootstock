"""Pack a built env into a single compressed image for node-local staging.

Cold worker starts on network filesystems are bounded by per-file metadata
RPCs and mmap fault storms (#167). A single ``tar.zst`` of the env tree
makes the transfer one sequential stream, and extracting it to node-local
disk (see :mod:`rootstock.stage`, #180) makes every subsequent ``stat``,
import, and mmap local. The image is an accelerator artifact derived from
the env — the Lustre tree stays the source of truth, and images are
regenerable at will.

Each image contains the env directory *and* the interpreter directory its
venv symlinks resolve through, both as root-relative paths::

    envs/<name>/...
    .python/cpython-3.11.15-linux-x86_64-gnu/...

so extraction reproduces the layout the venv expects. Weights are NOT
packed: they change on every ``rootstock add`` while envs change only on
``install``, so bundling them would stale the image constantly. They stage
separately from the manifest's per-checkpoint ``weight_files`` records.

Packing shells out to ``tar`` and ``zstd`` (probed up front) rather than
using Python archive modules: the same two tools are all extraction needs,
so a client can stage on any node where they exist, and zstd's multi-thread
compressor is far faster than anything in-process.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .exceptions import RootstockError

IMAGE_FORMAT = "tar.zst"
IMAGES_DIRNAME = "images"

# zstd level 3: within a few percent of higher levels on .so-heavy trees but
# several times faster to compress — and decompression speed (the spawn-path
# cost) is essentially level-independent.
_ZSTD_LEVEL = 3

# A .packing partial older than this is a crashed pack's leftover even if
# some process holds its recorded pid (pid reuse); far beyond any real pack.
_PACK_STALE_SECONDS = 6 * 3600.0


def _pid_alive(pid: int) -> bool:
    """Shared by the pack-side partial sweep and the staging module (which
    imports it from here — stage imports pack, so the helper can't live
    there without a cycle)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        pass
    return True


class PackError(RootstockError, RuntimeError):
    """Packing an env image failed. Messages are user-presentable."""


def pack_tools_missing() -> str | None:
    """Name the missing archive tool(s), or None when packing/staging can run."""
    missing = [tool for tool in ("tar", "zstd") if shutil.which(tool) is None]
    if missing:
        return " and ".join(missing)
    return None


def env_interpreter_dir(root: Path, env_name: str) -> Path:
    """The ``{root}/.python/<interpreter>`` directory this env's venv runs on.

    Resolved through ``envs/<name>/bin/python`` — the one link that must hold
    for the env to work at all. Raises PackError when it resolves outside the
    root's ``.python/`` (an env built against a foreign interpreter can't be
    made self-contained by this packer).
    """
    python_dir = (root / ".python").resolve()
    real_python = (root / "envs" / env_name / "bin" / "python").resolve()
    try:
        relative = real_python.relative_to(python_dir)
    except ValueError:
        raise PackError(
            f"env '{env_name}' runs on an interpreter outside {root / '.python'} "
            f"({real_python}) — cannot pack a self-contained image."
        ) from None
    return root / ".python" / relative.parts[0]


def _tree_bytes(*trees: Path) -> int:
    """Total apparent size of every regular file under ``trees`` (symlinks not
    followed). Runs right after a build, so the stats are metadata-cache warm."""
    total = 0
    for tree in trees:
        for dirpath, _dirnames, filenames in os.walk(tree):
            for filename in filenames:
                try:
                    total += os.lstat(os.path.join(dirpath, filename)).st_size
                except OSError:
                    continue
    return total


def pack_environment(root: Path | str, env_name: str, progress=None) -> dict:
    """Pack one built env into ``{root}/images/<name>-<sha12>.tar.zst``.

    Streams ``tar | zstd`` straight into the image file while hashing, so
    nothing is read back to compute the identity. The finished archive is
    renamed into place atomically and superseded images of the same env are
    removed. Returns the manifest ``image`` record (without ``packed_at``,
    which the manifest refresh stamps alongside ``built_at`` so the
    ``packed_at >= built_at`` currency check can't lose the race with its
    own install).

    Raises PackError when the env is not built, tools are missing, or the
    archive pipeline fails.
    """
    root = Path(root)
    env_dir = root / "envs" / env_name
    if not (env_dir / "bin" / "python").exists():
        raise PackError(f"env '{env_name}' is not built at {env_dir} — nothing to pack.")

    missing = pack_tools_missing()
    if missing:
        raise PackError(f"packing needs {missing} on PATH (on clusters, try `module load zstd`).")

    interp_dir = env_interpreter_dir(root, env_name)
    members = [
        str(env_dir.relative_to(root)),
        str(interp_dir.relative_to(root)),
    ]
    uncompressed = _tree_bytes(env_dir, interp_dir)

    images_dir = root / IMAGES_DIRNAME
    images_dir.mkdir(parents=True, exist_ok=True)
    partial = images_dir / f".{env_name}.packing.{os.getpid()}"

    if progress is not None:
        progress(f"  Packing {env_name}: {', '.join(members)} ({uncompressed / 1e9:.1f} GB)")

    digest = hashlib.sha256()
    compressed = 0
    tar_proc = zstd_proc = None
    try:
        with open(partial, "wb") as out:
            tar_proc = subprocess.Popen(
                ["tar", "-cf", "-", "-C", str(root), *members],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            zstd_proc = subprocess.Popen(
                ["zstd", f"-{_ZSTD_LEVEL}", "-T0", "-q", "-c"],
                stdin=tar_proc.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            assert tar_proc.stdout is not None and zstd_proc.stdout is not None
            tar_proc.stdout.close()  # so tar sees EPIPE if zstd dies
            while True:
                chunk = zstd_proc.stdout.read(1 << 20)
                if not chunk:
                    break
                digest.update(chunk)
                compressed += len(chunk)
                out.write(chunk)

        zstd_err = zstd_proc.communicate()[1]
        tar_err = tar_proc.communicate()[1]
        if tar_proc.returncode != 0:
            raise PackError(f"tar failed packing '{env_name}': {tar_err.decode(errors='replace')}")
        if zstd_proc.returncode != 0:
            raise PackError(
                f"zstd failed packing '{env_name}': {zstd_err.decode(errors='replace')}"
            )

        sha256 = digest.hexdigest()
        image_name = f"{env_name}-{sha256[:12]}.{IMAGE_FORMAT}"
        try:
            partial.rename(images_dir / image_name)
        except OSError as exc:
            # Most plausibly a concurrent pack of the same env swept our
            # partial; surface it as the domain error so batch callers
            # (pack_environments) report it instead of crashing.
            raise PackError(
                f"could not move the finished image for '{env_name}' into "
                f"place (concurrent pack?): {exc}"
            ) from exc
    except Exception:
        for proc in (tar_proc, zstd_proc):
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait()
        partial.unlink(missing_ok=True)
        raise

    # Superseded images of this env are dead weight on the shared
    # filesystem; the new archive is the only one the manifest will point
    # at. Exact-match on the <12-hex-sha> suffix — a bare `{env_name}-*`
    # glob would also swallow a dash-extended sibling env's images
    # (packing 'ani' must never delete 'ani-tuned-<sha>.tar.zst').
    stale_image = re.compile(rf"^{re.escape(env_name)}-[0-9a-f]{{12}}\.{re.escape(IMAGE_FORMAT)}$")
    for stale in images_dir.iterdir():
        if stale.name != image_name and stale_image.match(stale.name):
            stale.unlink(missing_ok=True)
    # Our own partial was renamed into place above; remaining .packing.*
    # entries are crashed packs' leftovers — unless their recorded pid is
    # still alive and the file is fresh (a concurrent pack of this env,
    # e.g. install's auto-pack racing a batch `rootstock pack`).
    for stale in images_dir.glob(f".{env_name}.packing.*"):
        try:
            pid = int(stale.name.rsplit(".", 1)[-1])
        except ValueError:
            pid = None
        try:
            age = time.time() - stale.stat().st_mtime
        except OSError:
            continue  # gone already
        if pid is not None and _pid_alive(pid) and age < _PACK_STALE_SECONDS:
            continue
        stale.unlink(missing_ok=True)

    if progress is not None:
        progress(
            f"  Packed {image_name}: {compressed / 1e9:.1f} GB (from {uncompressed / 1e9:.1f} GB)"
        )

    return {
        "path": f"{IMAGES_DIRNAME}/{image_name}",
        "sha256": sha256,
        "format": IMAGE_FORMAT,
        "compressed_bytes": compressed,
        "uncompressed_bytes": uncompressed,
    }


def pack_environment_best_effort(root: Path, env_name: str, progress=None) -> dict | None:
    """Pack, degrading to a warning: the image is an accelerator, so a failed
    pack (no zstd on PATH, say) must never fail the install that triggered it.
    The manifest's currency check then reports no usable image and spawns fall
    back to the prewarm path."""
    try:
        return pack_environment(root, env_name, progress=progress)
    except Exception as exc:  # noqa: BLE001 - packing is strictly optional
        print(
            f"Warning: could not pack an image for '{env_name}' "
            f"({exc}); worker spawns will use the prewarm path instead. "
            f"Retry later with `rootstock pack {env_name}`.",
            file=sys.stderr,
        )
        return None
