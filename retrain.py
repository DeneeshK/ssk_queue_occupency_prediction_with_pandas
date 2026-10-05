"""Simple retraining entry point.

Use the same training pipeline as train.py. The train.py promotion logic compares
validation MAE and MSE with the active model and does not replace it unless both improve.
"""

from train import main


if __name__ == "__main__":
    main()
