import os
import subprocess

import pytest


def test_cli_uses_env_var():
    root = "/tmp/fake-root"
    os.environ["ROOTSTOCK_ROOT"] = root
    out = subprocess.run(
        ["uv", "run", "rootstock", "status"],
        capture_output=True,
    )
    assert root in str(out.stdout)


def test_cli_root_overrides_env_var():
    env_root = "/tmp/env-root"
    os.environ["ROOTSTOCK_ROOT"] = env_root

    cli_root = "/tmp/cli-root"
    out = subprocess.run(
        ["uv", "run", "rootstock", "status", "--root", cli_root],
        capture_output=True,
    )
    assert env_root not in str(out.stdout)
    assert cli_root in str(out.stdout)
