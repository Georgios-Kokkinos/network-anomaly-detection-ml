"""
config.py

Central configuration for Version 4.
Adjust sampling, training, and output settings here.
"""

from dataclasses import dataclass


@dataclass
class Config:
    """Central configuration values for the Version 4 pipeline."""
    seed: int = 42
    label_col: str = "Label"

    # Data sampling (useful if memory is limited)
    sample_fraction: float = 1.0
    max_rows_per_file: int | None = 200_000
    drop_duplicate_rows: bool = True

    # Split sizes
    test_size: float = 0.2
    val_size: float = 0.1

    # Model/training
    batch_size: int = 1024
    epochs: int = 35
    learning_rate: float = 1e-4
    hidden_units: tuple[int, ...] = (512, 256, 128)
    dropout: float = 0.1
    use_class_weight: bool = True
    class_weight_power: float = 0.8
    max_class_weight: float = 80.0

    # Train-set balancing
    rebalance_train_data: bool = True
    min_train_samples_per_class: int = 1500
    max_train_samples_per_class: int = 100_000

    # Output
    results_dir_name: str = "results"
