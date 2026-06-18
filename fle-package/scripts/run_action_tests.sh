#!/usr/bin/env bash
# Run the Tower Defense action integration tests.
#
# Usage:
#   ./scripts/run_action_tests.sh               # start container, run, stop
#   ./scripts/run_action_tests.sh --no-start    # container already running; just run tests
#   ./scripts/run_action_tests.sh --no-stop     # keep the container alive after tests
#   ./scripts/run_action_tests.sh --no-start --no-stop  # both
#
# Extra pytest args are forwarded:
#   ./scripts/run_action_tests.sh -k test_resupply_turret
#   ./scripts/run_action_tests.sh --no-stop -s   # keep container + live stdout

set -euo pipefail
cd "$(dirname "$0")/.."

START_CONTAINER=true
STOP_CONTAINER=true
PYTEST_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --no-start) START_CONTAINER=false ;;
        --no-stop)  STOP_CONTAINER=false  ;;
        *)          PYTEST_ARGS+=("$arg") ;;
    esac
done

if $START_CONTAINER; then
    echo ">>> Starting Factorio container (default_lab_scenario)..."
    fle cluster start -n 1 -s default_lab_scenario
    echo ">>> Waiting for container to be ready..."
    sleep 5
fi

echo ">>> Running action integration tests..."
set +e
python3 -m pytest tests/integration/test_td_actions.py \
    -m integration \
    --log-cli-level=INFO \
    -v \
    "${PYTEST_ARGS[@]}"
EXIT_CODE=$?
set -e

if $STOP_CONTAINER; then
    echo ">>> Stopping container..."
    fle cluster stop
fi

exit $EXIT_CODE
