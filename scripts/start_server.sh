#!/bin/bash
# Start the Flower FL server with specified configuration

CONFIG="baseline"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: ./start_server.sh [--config <baseline|dp|krum>]"
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

echo "Starting Flower server with config: $CONFIG"

# Set active config in config.yaml
sed -i.bak "s/active_config:.*/active_config: \"$CONFIG\"/" config.yaml

# Start server
python -m server.server
