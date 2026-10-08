import argparse

from .pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Fit one or more photos to a printable body figure. One photo uses a "
        "geometric guess (the capsule/SDF builder); two or more (different angles around "
        "the subject) carve the actual shape via visual hull."
    )
    parser.add_argument(
        "--image",
        required=True,
        nargs="+",
        help="One photo, or several in turntable order (front first) for visual hull mode.",
    )
    parser.add_argument(
        "--angles",
        type=float,
        nargs="+",
        default=None,
        help="Rotation angle in degrees for each --image, same order. Only used with 2+ images. "
        "Defaults to front+side (0, 90) for two images, evenly spaced for three or more.",
    )
    parser.add_argument("--out", required=True, help="Output STL path.")
    parser.add_argument(
        "--height-mm",
        type=float,
        default=150.0,
        help="Target print height in millimeters (default: 150).",
    )
    args = parser.parse_args()

    run_pipeline(
        images=args.image,
        out_stl_path=args.out,
        target_height_mm=args.height_mm,
        angles_deg=args.angles,
    )


if __name__ == "__main__":
    main()
