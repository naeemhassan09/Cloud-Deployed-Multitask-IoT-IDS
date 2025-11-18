from pathlib import Path


def main():
    """
    Entry point for data preprocessing pipeline.

    Expected behaviour:
    - Read raw CIC IoT-IDAD 2024 data from data/raw
    - Apply cleaning, encoding, scaling
    - Write intermediate outputs to data/interim
    - Write final train/val/test splits to data/processed
    """
    raw_dir = Path("data/raw")
    interim_dir = Path("data/interim")
    processed_dir = Path("data/processed")

    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # TODO: implement preprocessing logic
    print(f"Raw data dir: {raw_dir}")
    print(f"Interim data dir: {interim_dir}")
    print(f"Processed data dir: {processed_dir}")


if __name__ == "__main__":
    main()
