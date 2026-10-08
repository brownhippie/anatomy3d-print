import argparse

from .pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Fit a photo to a stylized body figure and export a printable STL."
    )
    parser.add_argument("--image", required=True, help="Path to a front-facing, full-body photo.")
    parser.add_argument("--out", required=True, help="Output STL path.")
    parser.add_argument(
        "--height-mm",
        type=float,
        default=150.0,
        help="Target print height in millimeters (default: 150).",
    )
    args = parser.parse_args()

    run_pipeline(image_path=args.image, out_stl_path=args.out, target_height_mm=args.height_mm)


if __name__ == "__main__":
    main()
