import argparse
import os
import sys
import shutil
import subprocess
from pathlib import Path
import importlib.resources


def fle_init():
    if Path(".env").exists():
        return
    try:
        pkg = importlib.resources.files("fle")
        env_path = pkg / ".example.env"
        shutil.copy(str(env_path), ".env")
        print("Created .env file - please edit with your config")
    except Exception as e:
        print(f"Error during init: {e}", file=sys.stderr)
        sys.exit(1)


def fle_cluster(args):
    cluster_path = Path(__file__).parent / "cluster"
    script = cluster_path / "run-envs.sh"
    if not script.exists():
        print(f"Cluster script not found: {script}", file=sys.stderr)
        sys.exit(1)
    cmd = [str(script)]
    if args:
        if args.cluster_command:
            cmd.append(args.cluster_command)
        if args.n:
            cmd.extend(["-n", str(args.n)])
        if args.s:
            cmd.extend(["-s", args.s])
        if getattr(args, "save_path", None):
            cmd.extend(["--save-path", args.save_path])
    try:
        subprocess.run(cmd, cwd=str(cluster_path), check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running cluster script: {e}", file=sys.stderr)
        sys.exit(e.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Factorio Learning Environment CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  fle cluster start -n 4 -s default_lab_scenario
  fle cluster start -n 1 --save-path ./saves/my_map.zip
  fle cluster stop
        """,
    )
    subparsers = parser.add_subparsers(dest="command")

    # Cluster subcommand
    parser_cluster = subparsers.add_parser(
        "cluster", help="Setup Docker containers (run run-envs.sh)"
    )
    parser_cluster.add_argument(
        "cluster_command",
        nargs="?",
        choices=["start", "stop", "restart", "help"],
        help="Cluster command (start/stop/restart/help)",
    )
    parser_cluster.add_argument("-n", type=int, help="Number of Factorio instances")
    parser_cluster.add_argument(
        "-s",
        type=str,
        help="Scenario (open_world, default_lab_scenario, or tower_defense)",
    )
    parser_cluster.add_argument(
        "--save-path",
        type=str,
        help="Path to a Factorio save file (.zip) to use instead of a scenario",
    )

    # Init subcommand
    subparsers.add_parser("init", help="Initialize .env file")

    args = parser.parse_args()
    if args.command:
        fle_init()
    if args.command == "cluster":
        fle_cluster(args)
    elif args.command == "init":
        pass  # fle_init already ran above
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
