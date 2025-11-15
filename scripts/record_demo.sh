#!/bin/bash
# Record a demo run sequencing through all three configurations

echo "=== CER-based FL Demo Recording ==="
echo "This will run all three configurations sequentially."
echo ""

# Cleanup old logs and plots
rm -rf logs/*.jsonl plots/*.png

# Configurations to test
CONFIGS=("baseline" "dp" "krum")

for config in "${CONFIGS[@]}"; do
    echo ""
    echo "=== Running Configuration: $config ==="
    echo ""
    
    # Set active config
    sed -i.bak "s/active_config:.*/active_config: \"$config\"/" config.yaml
    
    # Start server in background
    python -m server.server &
    SERVER_PID=$!
    echo "Server started with PID $SERVER_PID"
    
    # Wait for server to initialize
    sleep 5
    
    # Start 3 clients for demo
    CLIENT_PIDS=()
    for i in 0 1 2; do
        echo "Starting client $i..."
        python -m clients.client --client-id $i &
        CLIENT_PIDS+=($!)
        sleep 1
    done
    
    echo "All clients started. Waiting for training to complete..."
    
    # Wait for server to complete
    wait $SERVER_PID
    
    # Kill any remaining clients
    for pid in ${CLIENT_PIDS[@]}; do
        kill $pid 2>/dev/null
    done
    
    echo "Configuration $config completed."
    sleep 3
done

echo ""
echo "=== Demo Complete ==="
echo "Generating plots..."

# Generate plots
python -m evaluate.plot

echo ""
echo "Demo recording complete!"
echo "- Logs: logs/rounds.jsonl"
echo "- Plots: plots/"
echo ""
echo "Launch dashboard with: streamlit run dashboard/app.py"
