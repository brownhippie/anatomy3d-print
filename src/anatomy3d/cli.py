import argparse

from .pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Fit a photo to an anatomically-plausible body model and export a printable STL."
    )
    parser.add_argument("--image", required=True, help="Path to a front-facing, full-body photo.")
    parser.add_argument(
        "--smplx-model-dir",
        required=True,
        help="Directory containing the downloaded SMPL-X model files.",
    )
    parser.add_argument("--gender", default="neutral", choices=["neutral", "male", "female"])
    parser.add_argument("--out", required=True, help="Output STL path.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    run_pipeline(
        image_path=args.image,
        smplx_model_dir=args.smplx_model_dir,
        out_stl_path=args.out,
        gender=args.gender,
        device=args.device,
    )


if __name__ == "__main__":
    main()
