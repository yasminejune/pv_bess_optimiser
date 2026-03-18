"""Entry point for running the inference package directly.

Allows: python -m ors.services.price_inference [args]
"""

from ors.services.price_inference.live_inference import main

main()
