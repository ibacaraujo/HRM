import os
import json
import numpy as np
from typing import Optional

from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm
from torchvision.datasets import CIFAR10

from common import PuzzleDatasetMetadata

cli = ArgParser()


class DataProcessConfig(BaseModel):
    root: str = "data/cifar10_raw"
    output_dir: str = "data/cifar10"
    train: Optional[bool] = None


def convert_subset(split: str, config: DataProcessConfig):
    dataset = CIFAR10(root=config.root, train=(split == "train"), download=True)

    inputs = []
    labels = []
    puzzle_indices = [0]
    group_indices = [0]
    puzzle_identifiers = []
    example_id = 0
    puzzle_id = 0

    for img, label in tqdm(dataset):
        arr = np.array(img, dtype=np.uint8)
        inputs.append(arr.transpose(2, 0, 1).reshape(-1) + 1)  # shift by 1 for PAD=0
        labels.append(np.array([label], dtype=np.int32))
        example_id += 1
        puzzle_id += 1
        puzzle_indices.append(example_id)
        puzzle_identifiers.append(0)
        group_indices.append(puzzle_id)

    results = {
        "inputs": np.stack(inputs),
        "labels": np.vstack(labels),
        "group_indices": np.array(group_indices, dtype=np.int32),
        "puzzle_indices": np.array(puzzle_indices, dtype=np.int32),
        "puzzle_identifiers": np.array(puzzle_identifiers, dtype=np.int32),
    }

    metadata = PuzzleDatasetMetadata(
        seq_len=32 * 32 * 3,
        vocab_size=256 + 1,
        pad_id=0,
        ignore_label_id=None,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(group_indices) - 1,
        mean_puzzle_examples=1,
        sets=["all"],
    )

    save_dir = os.path.join(config.output_dir, split)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)

    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    if config.train is None or config.train:
        convert_subset("train", config)
    if config.train is None or not config.train:
        convert_subset("test", config)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)


if __name__ == "__main__":
    cli()
