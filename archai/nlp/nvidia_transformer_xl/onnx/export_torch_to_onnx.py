# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse

from archai.nlp.nvidia_transformer_xl.onnx.onnx_utils.export import export_onnx_from_pt
from archai.nlp.nvidia_transformer_xl.onnx.onnx_utils.load import load_from_pt
from archai.nlp.nvidia_transformer_xl.onnx.onnx_utils.optimization import optimize_onnx
from archai.nlp.nvidia_transformer_xl.onnx.onnx_utils.quantization import dynamic_quantization


def parse_args():
    parser = argparse.ArgumentParser(description='Exports a PyTorch-based model to ONNX.')

    parser.add_argument('torch_model_path',
                        type=str,
                        help='Path to the PyTorch model/checkpoint file.')

    parser.add_argument('onnx_model_path',
                        type=str,
                        help='Path to the output ONNX model file.')

    parser.add_argument('-opt_level',
                        type=int,
                        default=0,
                        help='Level of the ORT optimization.')

    parser.add_argument('--optimization',
                        action='store_true',
                        help='Applies optimization to the exported model.')

    parser.add_argument('--quantization',
                        action='store_true',
                        help='Applies dynamic quantization to the exported model.')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # Gathers the command line arguments
    args = parse_args()

    # Transforms the command lines arguments into variables
    torch_model_path = args.torch_model_path
    onnx_model_path = args.onnx_model_path
    opt_level = args.opt_level
    optimization = args.optimization
    quantization = args.quantization

    # Loads the PyTorch model
    model, model_config = load_from_pt(torch_model_path)

    # Exports to ONNX
    export_onnx_from_pt(model, model_config, onnx_model_path, share_weights=True)

    # Whether optimization should be applied
    if optimization:
        ort_model_path = optimize_onnx(onnx_model_path, opt_level=opt_level)

        # Caveat to enable quantization after optimization
        onnx_model_path = ort_model_path

    # Whether dynamic quantization should be applied
    if quantization:
        qnt_model_path = dynamic_quantization(onnx_model_path)
