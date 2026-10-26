import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Export person ReID model to ONNX.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="weights/model.onnx",
        help="Output ONNX file path.",
    )
    args = parser.parse_args()
    print(f"[Export stub] Config: {args.config}, Output: {args.output}")


if __name__ == "__main__":
    main()
