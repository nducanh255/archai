# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
from pathlib import Path

from onnxruntime.quantization.quantize import quantize_dynamic


def parse_args():
    parser = argparse.ArgumentParser(description='Quantizes an ONNX model.')

    parser.add_argument('--path',
                        type=str,
                        help='Path to the pre-trained ONNX model file.')

    args = parser.parse_args()

    return args


def create_file_name_identifier(file_name: Path,
                                identifier: str) -> Path:
    """Adds a identifier (suffix) to the end of the file name.

    Args:
        file_name: Path to have a suffix added.
        identifier: Identifier to be added to file_name.

    Returns:
        (Path): Path with file_name plus added identifier.

    """

    return file_name.parent.joinpath(file_name.stem + identifier).with_suffix(file_name.suffix)


if __name__ == '__main__':
    # Gathers the command line arguments
    args = parse_args()

    # Performs the dynamic quantization
    qnt_model_path = create_file_name_identifier(Path(args.path), '-dynamically_quantized')
    quantize_dynamic(args.path, qnt_model_path)
