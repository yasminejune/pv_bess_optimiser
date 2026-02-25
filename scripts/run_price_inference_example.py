"""Example: run the live price inference pipeline and print the results.

Run from the project root::

    python scripts/run_price_inference_example.py

Optionally point at a different model::

    python scripts/run_price_inference_example.py --model models/price_prediction/my_model.pkl
"""

from __future__ import annotations

from ors.services.price_inference import run_inference


def main() -> None:
    return run_inference()
if __name__ == "__main__":
    main()
