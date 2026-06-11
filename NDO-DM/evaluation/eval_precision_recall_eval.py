import argparse
import numpy as np
import logging
from datetime import datetime
from pathlib import Path

from utils import seed_everthing, set_logger
from utils import compute_activation_and_logits
from utils import compute_activation_stat
from utils import compute_precision_recall  


def main(args):
    # Logging configuration
    target_dir = Path(args.output_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    time = datetime.now().strftime("%m%d-%H:%M:%S")
    set_logger(f=target_dir / f"pr_eval_{time}.log")

    # Print configuaration
    for key, val in vars(args).items():
        logging.info(f"{key}: {val}")
    seed_everthing(args.seed)

    # Load real data statistics (features)
    assert args.dataset in ["mnist", "fmnist", "celeba", "cifar10", "celeba64"]
    real_stat = np.load(f"_assets/{args.dataset}/stats.npz")
    real_features = real_stat["features"]  # shape: [N, D], e.g., 2048-dim from Inception

    # Load synthetic images
    synthesis = np.load(args.synthesis_path)
    images = synthesis["data"]

    assert isinstance(images, np.ndarray)
    assert len(images.shape) == 4

    if images.shape[1] == 3 or images.shape[1] == 1:
        images = np.transpose(images, [0, 2, 3, 1])
    if images.shape[3] == 1:
        images = images.repeat(3, axis=3)
    if images.dtype == np.uint8:
        images = images.astype(np.float32) / 255.0

    # Compute features of synthetic images
    logging.info("Computing synthetic image features...")
    fake_acts, _ = compute_activation_and_logits(images)  # [N, 2048]
    
    # Compute precision and recall
    logging.info("Computing precision and recall...")
    precision, recall = compute_precision_recall(real_features, fake_acts)

    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--synthesis_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="_results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    main(args)
