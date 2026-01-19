"""
config.py

Central configuration for Version 2.
Adjust sampling, training, and output settings here.
"""

from dataclasses import dataclass


@dataclass
class Config:
    """Central configuration values for the Version 2 pipeline."""
    seed: int = 42
    label_col: str = "Label"

    # Data sampling (useful if memory is limited)
    sample_fraction: float = 0.2
    max_rows_per_file: int | None = 200_000

    # Split sizes
    test_size: float = 0.2
    val_size: float = 0.1

    # Model/training
    batch_size: int = 1024
    epochs: int = 20
    learning_rate: float = 1e-3
    hidden_units: tuple[int, ...] = (256, 128, 64)
    dropout: float = 0.3
    use_class_weight: bool = True

    # Output
    results_dir_name: str = "results"
