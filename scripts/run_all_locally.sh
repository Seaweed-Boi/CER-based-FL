#!/bin/bash
# Run server and multiple clients locally

NUM_CLIENTS=3
CONFIG="baseline"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-clients)
            NUM_CLIENTS="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: ./run_all_locally.sh [--num-clients <N>] [--config <baseline|dp|krum>]"
            echo "  --num-clients: Number of clients to spawn (default: 3)"
            echo "  --config: FL configuration to use (default: baseline)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Starting FL server and $NUM_CLIENTS clients locally with config: $CONFIG"

# Set active config in config.yaml
sed -i.bak "s/active_config:.*/active_config: \"$CONFIG\"/" config.yaml

# Cleanup function
cleanup() {
    echo "Stopping all processes..."
    kill $SERVER_PID 2>/dev/null
    for pid in ${CLIENT_PIDS[@]}; do
        kill $pid 2>/dev/null
    done
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start server in background
python -m server.server &
SERVER_PID=$!
echo "Server started with PID $SERVER_PID"

# Wait for server to initialize
sleep 5

# Start clients
CLIENT_PIDS=()
for i in $(seq 0 $((NUM_CLIENTS-1))); do
    echo "Starting client $i..."
    python -m clients.client --client-id $i &
    CLIENT_PIDS+=($!)
    sleep 1
done

echo "All clients started. Press Ctrl+C to stop all processes."

# Wait for server to complete
wait $SERVER_PID
