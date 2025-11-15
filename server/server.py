"""
Flower FL Server for CER-based FL experiments.
Orchestrates federated learning rounds and aggregates client updates.

Usage:
    python -m server.server --config config.yaml
"""
import argparse
import yaml
import flwr as fl
from server.logger import RoundLogger
from server.strategy_custom import get_strategy


def main():
    parser = argparse.ArgumentParser(description="Start Flower FL Server")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Get active configuration
    active_config_name = config.get("active_config", "baseline")
    print(f"\n=== Starting FL Server ===")
    print(f"Active configuration: {active_config_name}")
    
    # Initialize logger
    logger = RoundLogger(
        config["logging"]["output_file"],
        config_name=active_config_name
    )

    # Get strategy
    strategy = get_strategy(config, logger)

    # Start server
    server_address = config["server"]["address"]
    num_rounds = config["server"]["num_rounds"]

    print(f"Server address: {server_address}")
    print(f"Number of rounds: {num_rounds}")
    print(f"Aggregation method: {config['configs'][active_config_name]['agg_method']}")
    print("Waiting for clients...")
    print("="*40 + "\n")

    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    logger.close()
    print("\n=== Server finished ===")


if __name__ == "__main__":
    main()
