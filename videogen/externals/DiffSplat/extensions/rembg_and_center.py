import os
import glob
import argparse
import logging

import numpy as np
import cv2
import rembg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove background and center the image of an object"
    )

    parser.add_argument(
        "dir_or_path",
        type=str,
        help="Directory or path to images (png, jpeg, webp, etc.)"
    )
    parser.add_argument(
        "--model_name",
        default="u2net",  # "isnet-general-use", "birefnet-general", "birefnet-dis", "birefnet-massive"
        type=str,
        help="Rembg model, see https://github.com/danielgatis/rembg#models"
    )
    parser.add_argument(
        "--size",
        default=512,
        type=int,
        help="Output resolution"
    )
    parser.add_argument(
        "--border_ratio",
        default=0.2,
        type=float,
        help="Output border ratio"
    )
    parser.add_argument(
        "--center",
        action="store_true",
        help="Center the object, potentially not helpful for multiview zero123"
    )

    # Parse the arguments
    args = parser.parse_args()

    # Initialize the logger
    logging.basicConfig(
        format="%(asctime)s - REMBG&CENTER - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO
    )
    logger = logging.getLogger(__name__)
    logger.propagate = True  # propagate to the root logger (console)

    # Create a session for rembg
    session = rembg.new_session(model_name=args.model_name)

    if os.path.isdir(args.dir_or_path):
        logger.info(f"Processing directory [{args.dir_or_path}]...")
        files = glob.glob(f"{args.dir_or_path}/*")
        out_dir = args.dir_or_path
    else:  # single file
        files = [args.dir_or_path]
        out_dir = os.path.dirname(args.dir_or_path)

    for file in files:
        out_base = os.path.basename(file).split(".")[0]
        out_rgba = os.path.join(out_dir, out_base + "_rgba.png")

        # Load image and resize
        logger.info(f"Loading image [{file}]...")
        image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        _h, _w = image.shape[:2]
        scale = args.size / max(_h, _w)
        _h, _w = int(_h * scale), int(_w * scale)
        image = cv2.resize(image, (_w, _h), interpolation=cv2.INTER_AREA)

        # Remove background
        logger.info("Removing background...")
        carved_image = rembg.remove(image, session=session) # (H, W, 4)
        mask = carved_image[..., -1] > 0

        # Center the object
        if args.center:
            logger.info("Centering object...")
            final_rgba = np.zeros((args.size, args.size, 4), dtype=np.uint8)

            coords = np.nonzero(mask)
            x_min, x_max = coords[0].min(), coords[0].max()
            y_min, y_max = coords[1].min(), coords[1].max()
            h = x_max - x_min
            w = y_max - y_min
            desired_size = int(args.size * (1 - args.border_ratio))
            scale = desired_size / max(h, w)
            h2 = int(h * scale)
            w2 = int(w * scale)
            x2_min = (args.size - h2) // 2
            x2_max = x2_min + h2
            y2_min = (args.size - w2) // 2
            y2_max = y2_min + w2
            final_rgba[x2_min:x2_max, y2_min:y2_max] = cv2.resize(
                carved_image[x_min:x_max, y_min:y_max],
                (w2, h2),
                interpolation=cv2.INTER_AREA
            )
        else:
            final_rgba = carved_image
        
        # Save image
        cv2.imwrite(out_rgba, final_rgba)

    print()  # newline after the process
