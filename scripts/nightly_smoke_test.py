# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ['globus-compute-sdk<4.7', 'rootstock']
#
# [tool.uv]
# python-preference = "managed"
#
# [tool.hog.delta]
# endpoint = "27687af7-a20e-477a-8d4e-b5a7a097f864"
# account = "bhhl-delta-gpu"                              # Type: string
# scheduler_options = "#SBATCH --gpus-per-node=1"         # Type: string
# # qos =                                                 # Type: string
# # walltime =                                            # Type: string
# # exclusive =                                           # Type: boolean
# # partition =                                           # Type: string
# # constraint =                                          # Type: string
# # max_blocks =                                          # Type: integer
# # min_blocks =                                          # Type: integer
# # init_blocks =                                         # Type: integer
# # mem_per_node =                                        # Type: integer
# # cores_per_node =                                      # Type: integer
# # endpoint_setup =                                      # Type: string
# # max_workers_per_node =                                # Type: integer
# ///

import sys

import groundhog_hpc as hog


@hog.function(endpoint="delta")
def smoke_test(root: str | None = None) -> dict:
    import subprocess

    cmd = ["rootstock", "smoke-test", "--device", "cuda", "--json"]
    if root is not None:
        cmd += ["--root", root]

    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3000)
    return {"returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr}


@hog.function(endpoint="delta")
def hello_endpoint():
    import getpass
    import platform
    import socket
    import time

    return {
        "hostname": socket.gethostname(),
        "user": getpass.getuser(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "epoch": time.time(),
    }


@hog.harness()
def hello(endpoint: str = "delta"):
    from pprint import pprint

    result = hello_endpoint.remote(endpoint=endpoint)
    pprint(result)
    return


@hog.harness()
def main(target: str | None = None, root: str | None = None) -> int:
    if target:
        fut = smoke_test.submit(root, endpoint=target)
    else:
        fut = smoke_test.submit(root)

    result = fut.result()

    print(result["stdout"])
    print(result["stderr"], file=sys.stderr)

    # 0 = all passed; 1 = some failed but manifest was updated. Both mean the
    # deliverable (fresh manifest) succeeded. Anything else is a transport
    # failure and should turn CI red.
    rc = result["returncode"]
    return 0 if rc in (0, 1) else 2
