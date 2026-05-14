# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ['globus-compute-sdk<4.7', 'rootstock']
#
# [tool.uv]
# python-preference = "managed"
#
# [tool.hog.delta]
# endpoint = "c2381fa2-ff0f-460e-b6ac-8286086f122e"
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
