"""Generate synthetic patient appointment data for no-show prediction.

Produces a CSV at `data/patients.csv` by default.

Features:
- lead_time (days between booking and appointment)
- distance_km (patient travel distance)
- past_no_shows (count of prior no-shows)
- age
- gender
- appointment_type
- no_show (0/1 target)

Clean, testable functions and a CLI.
"""
from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def generate_synthetic_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic patient dataset for no-show modeling.

    Args:
        n_samples: Number of rows to generate.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with features and target `no_show`.
    """
    rng = np.random.default_rng(seed)

    # Numerical features
    lead_time = rng.integers(0, 120, size=n_samples)  # days
    distance_km = np.round(rng.exponential(scale=5, size=n_samples), 2)
    past_no_shows = rng.integers(0, 10, size=n_samples)
    age = rng.integers(0, 100, size=n_samples)

    # Categorical features
    gender = rng.choice(["M", "F", "Other"], size=n_samples, p=[0.48, 0.48, 0.04])
    appointment_type = rng.choice(
        ["PrimaryCare", "Specialist", "Lab", "Imaging"], size=n_samples, p=[0.6, 0.25, 0.1, 0.05]
    )
    weather = rng.choice(["Sunny", "Cloudy", "Rainy", "Stormy"], size=n_samples, p=[0.6, 0.2, 0.15, 0.05])

    # Risk score: linear combination + non-linear effects
    score = (
        0.05 * lead_time
        + 0.5 * (distance_km / (distance_km + 1))
        + 1.0 * (past_no_shows)
        + 0.05 * np.maximum(0, (age - 60))
    )

    # Small appointment-type and gender effects
    score += np.where(appointment_type == "Specialist", 0.2, 0.0)
    score += np.where(gender == "Other", 0.15, 0.0)
    
    # Weather effects
    score += np.where(weather == "Rainy", 0.5, 0.0)
    score += np.where(weather == "Stormy", 1.5, 0.0)
    # Extra penalty for distance in storms
    score += np.where((weather == "Stormy") & (distance_km > 10), 0.5, 0.0)

    # Convert to probability and sample binary target
    prob_no_show = _sigmoid(score - np.median(score))  # center scores
    no_show = rng.binomial(1, prob_no_show)

    df = pd.DataFrame(
        {
            "lead_time": lead_time,
            "distance_km": distance_km,
            "past_no_shows": past_no_shows,
            "age": age,
            "gender": gender,
            "appointment_type": appointment_type,
            "weather": weather,
            "no_show": no_show,
            # Admin features
            "appointment_mode": np.random.choice(["Offline", "Online"], n_samples, p=[0.8, 0.2]),
        }
    )

    return df


def save_dataset(df: pd.DataFrame, path: str = "data/patients.csv") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def main(args: argparse.Namespace) -> None:
    df = generate_synthetic_data(n_samples=args.n_samples, seed=args.seed)
    save_dataset(df, path=args.output)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic patient data for JijaArogyaCare")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="data/patients.csv")
    args = parser.parse_args()
    main(args)
