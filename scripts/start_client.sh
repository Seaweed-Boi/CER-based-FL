#!/bin/bash
# Start a single Flower FL client

CLIENT_ID=0
CONFIG="baseline"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --client-id)
            CLIENT_ID="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: ./start_client.sh [--client-id <id>] [--config <baseline|dp|krum>]"
            echo "  --client-id: Client ID (default: 0)"
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

echo "Starting Flower client $CLIENT_ID with config: $CONFIG"

# Set active config in config.yaml
sed -i.bak "s/active_config:.*/active_config: \"$CONFIG\"/" config.yaml

# Start client
python -m clients.client --client-id $CLIENT_ID
