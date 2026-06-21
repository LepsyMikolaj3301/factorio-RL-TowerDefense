import argparse
import os
import sys
import shutil
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
    # Route through the Python ClusterManager (handles tower_defense config
    # selection and save-file loading, which the legacy shell script does not).
    from fle.cluster.run_envs import start_cluster, stop_cluster, restart_cluster

    command = (args.cluster_command if args else None) or "start"

    if command == "stop":
        stop_cluster()
        return
    if command == "restart":
        restart_cluster()
        return
    if command == "help":
        print(
            "Usage: fle cluster [start|stop|restart] [-n N] [-s SCENARIO] "
            "[--save-path FILE] [--generate]\n"
            "  tower_defense loads data/saves/tower_defense.zip by default; "
            "pass --generate to build a fresh map."
        )
        return

    # start
    num_instances = args.n or 1
    scenario = args.s or "tower_defense"
    save_path = getattr(args, "save_path", None)
    generate = getattr(args, "generate", False)
    try:
        start_cluster(
            num_instances=num_instances,
            scenario=scenario,
            save_file=save_path,
            generate=generate,
        )
    except SystemExit:
        raise
    except Exception as e:
        print(f"Error starting cluster: {e}", file=sys.stderr)
        sys.exit(1)


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
        help="Scenario (open_world, default_lab_scenario, or tower_defense). "
        "Defaults to tower_defense.",
    )
    parser_cluster.add_argument(
        "--save-path",
        type=str,
        help="Path to a Factorio save file (.zip) to load. For tower_defense, "
        "defaults to data/saves/tower_defense.zip.",
    )
    parser_cluster.add_argument(
        "--generate",
        action="store_true",
        help="Generate a fresh map from the scenario instead of loading a save. "
        "Not supported for tower_defense, which always requires a predefined save.",
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
