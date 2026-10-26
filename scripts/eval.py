import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate open-set person ReID model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()
    print(f"[Evaluation stub] Config: {args.config}")


if __name__ == "__main__":
    main()
