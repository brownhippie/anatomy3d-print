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
        "Defaults to front+side (0, 90) for two images, front+both sides (0, 90, 270) "
        "for three, evenly spaced for four or more.",
    )
    parser.add_argument("--out", required=True, help="Output STL path.")
    parser.add_argument(
        "--height-mm",
        type=float,
        default=150.0,
        help="Target print height in millimeters (default: 150).",
    )
    parser.add_argument(
        "--use-depth",
        action="store_true",
        help="Single-photo mode only: sculpt the front surface with real per-pixel depth "
        "(Depth Anything V2 Small) instead of the flat symmetric guess. Needs "
        "`pip install -r requirements-depth.txt`.",
    )
    parser.add_argument(
        "--target-faces",
        type=int,
        default=20000,
        help="Simplify the output STL to roughly this many faces after repair (default: "
        "20000). The OBJ is unaffected and stays full-detail. Pass 0 to export at full "
        "detail with no simplification.",
    )
    args = parser.parse_args()

    run_pipeline(
        images=args.image,
        out_stl_path=args.out,
        target_height_mm=args.height_mm,
        angles_deg=args.angles,
        use_depth=args.use_depth,
        target_faces=args.target_faces or None,
    )


if __name__ == "__main__":
    main()
