# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ['globus-compute-sdk<4.7', 'rootstock']
#
# [tool.uv]
# exclude-newer = "2026-05-14T20:19:30Z"
# python-preference = "managed"
#
# [tool.hog.delta]
# endpoint = "67a5485a-4084-41f0-9863-2d5de388276e"
# ///

import sys

import groundhog_hpc as hog


@hog.function()
def smoke_test(root: str | None = None) -> dict:
    import subprocess

    cmd = ["rootstock", "smoke-test", "--device", "cuda", "--json"]
    if root is not None:
        cmd += ["--root", root]

    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3000)
    return {"returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr}


@hog.harness()
def main(target: str = "delta", root: str | None = None) -> int:
    result = smoke_test.remote(root, endpoint=target)

    print(result["stdout"])
    print(result["stderr"], file=sys.stderr)

    # 0 = all passed; 1 = some failed but manifest was updated. Both mean the
    # deliverable (fresh manifest) succeeded. Anything else is a transport
    # failure and should turn CI red.
    rc = result["returncode"]
    return 0 if rc in (0, 1) else 2
