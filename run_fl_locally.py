"""
Run FL server and clients locally on Windows.
Starts server and multiple clients as separate processes.
"""
import subprocess
import time
import signal
import sys
import yaml
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Run FL locally")
    parser.add_argument("--num-clients", type=int, default=3, help="Number of clients")
    parser.add_argument("--config", type=str, default="baseline", help="Config name (baseline, dp, krum)")
    args = parser.parse_args()
    
    print(f"Starting FL with {args.num_clients} clients, config: {args.config}")
    
    # Update config.yaml with active config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    config["active_config"] = args.config
    
    with open("config.yaml", "w") as f:
        yaml.safe_dump(config, f)
    
    print(f"Set active_config to: {args.config}")
    
    # Set environment variable to use pure Python protobuf
    env = os.environ.copy()
    env["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    
    processes = []
    
    try:
        # Start server
        print("\nStarting server...")
        server_proc = subprocess.Popen(
            [sys.executable, "-m", "server.server"],
            env=env,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        )
        processes.append(server_proc)
        print(f"Server started (PID: {server_proc.pid})")
        
        # Wait for server to initialize
        print("Waiting for server to initialize...")
        time.sleep(5)
        
        # Start clients
        for i in range(args.num_clients):
            print(f"Starting client {i}...")
            client_proc = subprocess.Popen(
                [sys.executable, "-m", "clients.client", "--client-id", str(i)],
                env=env,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
            processes.append(client_proc)
            print(f"Client {i} started (PID: {client_proc.pid})")
            time.sleep(1)
        
        print("\nAll processes started. Waiting for completion...")
        print("Press Ctrl+C to stop all processes.\n")
        
        # Wait for server to complete
        server_proc.wait()
        
        print("\n=== FL Training Complete ===")
        
    except KeyboardInterrupt:
        print("\n\nStopping all processes...")
    finally:
        # Cleanup: terminate all processes
        for proc in processes:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except:
                try:
                    proc.kill()
                except:
                    pass
        print("All processes stopped.")

if __name__ == "__main__":
    main()
