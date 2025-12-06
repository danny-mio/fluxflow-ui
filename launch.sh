#!/bin/bash
# FluxFlow UI - Launch Script
# Launches the web interface

set -e

# Default settings
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5000}"
DEBUG="${DEBUG:-false}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --debug)
            DEBUG="true"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=== FluxFlow UI ==="
echo "Host: $HOST"
echo "Port: $PORT"
echo "Debug: $DEBUG"
echo ""
echo "Opening http://$HOST:$PORT"
echo ""

# Build command
CMD="fluxflow-ui --host $HOST --port $PORT"
if [ "$DEBUG" == "true" ]; then
    CMD="$CMD --debug"
fi

# Run
$CMD
