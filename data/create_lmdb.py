"""Convert a folder of images in the same directory to lmdb."""

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(project_dir)

print("Script directory:", script_dir)
print("Project directory:", project_dir)
print("sys.path:", sys.path)

import click
import logging
from datasets import lmdb_dataset


script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(project_dir)


@click.command()
@click.option("--dpath", required=True, help="Path to the folder containing images.")
@click.option(
    "--split", default="train", help="Dataset split (e.g., train, val, test)."
)
@click.option("--num-workers", type=int, default=0, help="Number of worker processes.")
def main(dpath, split, num_workers):
    try:
        logging.info(
            f"Starting conversion: {dpath} to lmdb with split '{split}' using {num_workers} workers."
        )
        lmdb_dataset.folder2lmdb(dpath, split, num_workers)
        logging.info("Conversion completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred during conversion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
