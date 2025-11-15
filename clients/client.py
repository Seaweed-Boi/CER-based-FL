"""
Flower FL Client for binary fraud classification.
Handles local training with DP and attack options.

Usage:
    python -m clients.client --client-id 0 --config config.yaml
"""
import argparse
import yaml
import flwr as fl
import torch
import numpy as np
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from clients.model import create_model
from clients.train_loop import train, evaluate
from data.preprocess import load_client_data, load_test_data


class FlowerClient(fl.client.NumPyClient):
    """Flower client for federated fraud detection."""

    def __init__(self, client_id: int, config: dict, model, trainloader, testloader):
        self.client_id = client_id
        self.config = config
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Determine if this client is malicious
        active_config_name = config.get("active_config", "baseline")
        active_config = config["configs"][active_config_name]
        self.is_malicious = client_id in active_config.get("attack_clients", [])
        self.use_dp = active_config.get("use_dp", False)
        
        if self.is_malicious:
            print(f"Client {client_id}: MALICIOUS (will perform attacks)")
        if self.use_dp:
            print(f"Client {client_id}: DP enabled")

    def get_parameters(self, config):
        """Return model parameters as list of NumPy arrays."""
        return [val.cpu().detach().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Update model parameters from list of NumPy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Train model on local data."""
        print(f"\nClient {self.client_id}: Starting local training...")
        
        self.set_parameters(parameters)
        
        # Get training config
        client_config = self.config["clients"]
        epochs = client_config["epochs_per_round"]
        lr = client_config["learning_rate"]
        
        # DP config
        dp_config = None
        if self.use_dp:
            active_config_name = self.config.get("active_config")
            dp_config = self.config["configs"][active_config_name]
        
        # Attack config
        attack_config = self.config.get("attack", {}) if self.is_malicious else None
        
        # Train
        self.model, privacy_engine = train(
            self.model,
            self.trainloader,
            epochs=epochs,
            device=self.device,
            lr=lr,
            use_dp=self.use_dp,
            dp_config=dp_config,
            is_malicious=self.is_malicious,
            attack_config=attack_config
        )
        
        print(f"Client {self.client_id}: Training complete")
        
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        """Evaluate model on local test data."""
        self.set_parameters(parameters)
        y_true, y_pred = evaluate(self.model, self.testloader, device=self.device)
        
        # Compute basic accuracy
        accuracy = np.mean(np.array(y_true) == np.array(y_pred))
        
        return 0.0, len(self.testloader.dataset), {"accuracy": float(accuracy)}


def main():
    parser = argparse.ArgumentParser(description="Start Flower FL Client")
    parser.add_argument("--client-id", type=int, required=True, help="Client ID")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print(f"\n=== Client {args.client_id} ===")

    # Load client data
    batch_size = config["clients"]["batch_size"]
    trainloader = load_client_data(args.client_id, batch_size)
    testloader = load_test_data(batch_size)
    
    print(f"Loaded {len(trainloader.dataset)} training samples")
    print(f"Loaded {len(testloader.dataset)} test samples")

    # Create model (determine input_dim from data)
    sample_features, _ = next(iter(trainloader))
    input_dim = sample_features.shape[1]
    model = create_model(input_dim=input_dim)
    
    print(f"Model created with input_dim={input_dim}")

    # Create client
    client = FlowerClient(args.client_id, config, model, trainloader, testloader)

    # Connect to server
    server_address = config["server"]["address"]
    print(f"Connecting to server at {server_address}...")

    fl.client.start_numpy_client(server_address=server_address, client=client)
    
    print(f"Client {args.client_id}: Disconnected")


if __name__ == "__main__":
    main()
