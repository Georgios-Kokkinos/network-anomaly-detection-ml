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
    sample_fraction: float = 0.8
    max_rows_per_file: int | None = 200_000

    # Split sizes
    test_size: float = 0.2
    val_size: float = 0.1

    # Model/training
    batch_size: int = 512
    epochs: int = 14
    learning_rate: float = 1e-3
    hidden_units: tuple[int, ...] = (512, 256, 128)
    dropout: float = 0.2
    use_class_weight: bool = False

    # Train-set balancing
    rebalance_train_data: bool = True
    min_train_samples_per_class: int = 1500
    max_train_samples_per_class: int = 40_000

    # Output
    results_dir_name: str = "results"
