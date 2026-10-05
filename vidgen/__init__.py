"""vidgen — a free text-to-video generator.

The heavy lifting runs on a borrowed GPU (a free Colab / Kaggle notebook, or any
GPU pod) inside ``vidgen.server``; you drive it from your own machine with the
CLI (``python -m vidgen``) or the local web UI (``python -m vidgen ui``), and the
finished videos are saved locally.
"""

__version__ = "0.1.0"
