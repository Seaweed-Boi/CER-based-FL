"""
Logger for FL rounds - writes metrics to JSONL file with proper schema.

Schema: {"round": int, "config": str, "F2": float, "epsilon": float|null, 
         "comm_kb": float, "resilience": float, "CER": float}

Usage:
    from server.logger import RoundLogger
"""
import json
import time
from typing import Dict, Any, Optional
from pathlib import Path


class RoundLogger:
    """Logs FL round metrics to JSONL file with proper schema."""

    def __init__(self, log_file: str, config_name: str):
        """
        Args:
            log_file: Path to log file
            config_name: Name of active configuration (baseline, dp, krum)
        """
        self.log_file = Path(log_file)
        self.config_name = config_name
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create/clear log file
        with open(self.log_file, "a") as f:
            pass
        
        print(f"Logger initialized: {log_file} (config: {config_name})")

    def log_round(self, round_num: int, f2_score: float, epsilon: Optional[float], 
                  comm_kb: float, resilience: float, cer: float):
        """
        Log metrics for a single round with proper schema.
        
        Args:
            round_num: Round number
            f2_score: F2 score
            epsilon: Privacy budget (or None if DP off)
            comm_kb: Communication cost in KB
            resilience: Resilience metric
            cer: CER score
        """
        log_entry = {
            "round": round_num,
            "config": self.config_name,
            "F2": float(f2_score),
            "epsilon": float(epsilon) if epsilon is not None else None,
            "comm_kb": float(comm_kb),
            "resilience": float(resilience),
            "CER": float(cer),
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def close(self):
        """Close logger (placeholder for future cleanup)."""
        pass
