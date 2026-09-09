import argparse
import subprocess
from pathlib import Path
import yaml


def load_yaml(path):
    path = Path(path)

    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_prediction(config_path):
    config = load_yaml(config_path)

    dataset_id = str(config["dataset"]["id"])
    configuration = config["model"]["configuration"]

    input_folder = config["prediction"]["input_folder"]
    output_folder = Path(config["prediction"]["output_folder"])

    output_folder.mkdir(parents=True, exist_ok=True)

    cmd = [
        "nnUNetv2_predict",
        "-i", input_folder,
        "-o", str(output_folder),
        "-d", dataset_id,
        "-c", configuration,
    ]

    print("\nRunning ensemble prediction:")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run nnU-Net prediction from YAML config"
    )

    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="Path to prediction YAML file"
    )

    args = parser.parse_args()

    run_prediction(args.config)