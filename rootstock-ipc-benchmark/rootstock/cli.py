"""
Rootstock CLI.

Commands:
    rootstock test <env_file> --model <model> [--device <device>] [--root <path>]
    rootstock register <env_file> --root <path>
    rootstock list --root <path>
"""

import argparse
import shutil
import sys
import time
from pathlib import Path


def cmd_test(args) -> int:
    """
    Test an environment file.

    Exit codes:
        0: Success
        1: Environment creation failed
        2: Import/setup failed
        3: Calculation failed
    """
    from .environment import check_uv_available
    from .pep723 import validate_environment_file

    env_path = Path(args.env_file)

    # Step 1: Validate the environment file
    print(f"Validating {env_path}...")
    is_valid, error = validate_environment_file(env_path)
    if not is_valid:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    print("  PEP 723 metadata: OK")
    print("  setup() function: OK")

    # Step 2: Check uv is available
    if not check_uv_available():
        print(
            "Error: uv not found in PATH. Install uv: "
            "https://docs.astral.sh/uv/getting-started/installation/",
            file=sys.stderr,
        )
        return 1

    # Step 3: Set up root directory
    root = Path(args.root) if args.root else None
    if root:
        root.mkdir(parents=True, exist_ok=True)

    # Step 4: Create test system
    print("\nCreating test system...")
    try:
        from ase.build import bulk

        atoms = bulk("Cu", "fcc", a=3.6) * (2, 2, 2)  # 8 atoms
        print(f"  Created Cu FCC system: {len(atoms)} atoms")
    except ImportError:
        print("Error: ASE not installed", file=sys.stderr)
        return 1

    # Step 5: Run calculation through RootstockCalculator
    print(f"\nTesting environment with model={args.model}, device={args.device}...")

    try:
        from .calculator import RootstockCalculator

        start_time = time.time()

        with RootstockCalculator(
            environment=str(env_path.absolute()),
            model=args.model,
            device=args.device,
            root=root,
            log=sys.stderr if args.verbose else None,
        ) as calc:
            atoms.calc = calc

            # Warmup
            print("  Warming up...")
            _ = atoms.get_potential_energy()

            # Timed calculation
            print("  Running calculation...")
            calc_start = time.time()
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            calc_time = time.time() - calc_start

        total_time = time.time() - start_time

        print(f"\n  Energy: {energy:.6f} eV")
        print(f"  Forces shape: {forces.shape}")
        print(f"  Max force: {abs(forces).max():.6f} eV/A")
        print(f"\n  Calculation time: {calc_time*1000:.1f} ms")
        print(f"  Total time (incl. setup): {total_time:.1f} s")

        print("\nTest PASSED")
        return 0

    except Exception as e:
        print(f"\nError during calculation: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 3


def cmd_register(args) -> int:
    """Register an environment file to the shared directory."""
    from .pep723 import validate_environment_file

    env_path = Path(args.env_file)
    root = Path(args.root)

    # Validate the file
    print(f"Validating {env_path}...")
    is_valid, error = validate_environment_file(env_path)
    if not is_valid:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    # Create environments directory
    env_dir = root / "environments"
    env_dir.mkdir(parents=True, exist_ok=True)

    # Copy file
    dest_path = env_dir / env_path.name
    shutil.copy2(env_path, dest_path)

    print(f"Registered: {env_path.stem} -> {dest_path}")
    return 0


def cmd_list(args) -> int:
    """List registered environments."""
    from .environment import list_environments

    root = Path(args.root)

    environments = list_environments(root)

    if not environments:
        print(f"No environments registered in {root}")
        return 0

    print(f"Registered environments in {root}:")
    for name, path in environments:
        print(f"  {name:<15} {path}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="rootstock",
        description="Rootstock MLIP environment manager",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # test command
    test_parser = subparsers.add_parser(
        "test",
        help="Test an environment file",
        description="Test that an environment file works correctly.",
    )
    test_parser.add_argument("env_file", help="Path to environment file")
    test_parser.add_argument("--model", required=True, help="Model identifier (e.g., mace-mp-0)")
    test_parser.add_argument("--device", default="cuda", help="Device string (default: cuda)")
    test_parser.add_argument("--root", help="Root directory for cache")
    test_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    test_parser.set_defaults(func=cmd_test)

    # register command
    reg_parser = subparsers.add_parser(
        "register",
        help="Register an environment file",
        description="Copy a validated environment file to the shared environments directory.",
    )
    reg_parser.add_argument("env_file", help="Path to environment file")
    reg_parser.add_argument("--root", required=True, help="Root directory")
    reg_parser.set_defaults(func=cmd_register)

    # list command
    list_parser = subparsers.add_parser(
        "list",
        help="List registered environments",
        description="List all environment files in the shared environments directory.",
    )
    list_parser.add_argument("--root", required=True, help="Root directory")
    list_parser.set_defaults(func=cmd_list)

    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
