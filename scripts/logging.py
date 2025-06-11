"""
scripts/logging.py

Centralized logging and result-directory utilities for NeuroPRIN:
- create_results_directory: make timestamped run folder
- save_json_config: write configuration dict to JSON
- log_model_result: append model metrics to CSV or write JSON summary
"""
import os
import json
import csv
from datetime import datetime
from typing import Any, Dict


def create_results_directory(
    root_dir: str = 'results',
    run_name: str = None
) -> str:
    """
    Create and return a new results directory under root_dir.
    If run_name is None, use timestamp. Returns full path.
    """
    os.makedirs(root_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    folder_name = run_name or f'run_{timestamp}'
    run_dir = os.path.join(root_dir, folder_name)
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


def save_json_config(
    filepath: str,
    config: Dict[str, Any]
) -> None:
    """
    Save a configuration dictionary to a JSON file at filepath.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)


def log_model_result(
    results_dir: str,
    model_name: str,
    metrics: Dict[str, Any],
    filename: str = 'metrics.csv'
) -> None:
    """
    Log model metrics to a CSV file in results_dir. If the file exists, append; else create with header.
    metrics: Dict of metric_name -> value
    """
    filepath = os.path.join(results_dir, filename)
    file_exists = os.path.isfile(filepath)
    with open(filepath, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['model'] + list(metrics.keys()))
        if not file_exists:
            writer.writeheader()
        row = {'model': model_name}
        row.update(metrics)
        writer.writerow(row)


def log_run_summary(
    results_dir: str,
    summary: Dict[str, Any],
    filename: str = 'summary.json'
) -> None:
    """
    Write run summary dictionary to JSON in results_dir.
    """
    filepath = os.path.join(results_dir, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(summary, f, indent=2)
