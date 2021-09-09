# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from os import environ

# Constants available in onnxruntime
# that enables performance optimization
environ["OMP_NUM_THREADS"] = str(1)
environ["OMP_WAIT_POLICY"] = 'ACTIVE'

import argparse

import torch
from onnxruntime import (GraphOptimizationLevel, InferenceSession,
                         SessionOptions)


def parse_args():
    parser = argparse.ArgumentParser(description='Compares logits between ONNX models.')

    parser.add_argument('onnx_model_path',
                        type=str,
                        help='Path to the pre-trained ONNX model file.')

    parser.add_argument('qnt_onnx_model_path',
                        type=str,
                        help='Path to the pre-trained ONNX quantized model file.')

    parser.add_argument('-batch_size',
                        type=int,
                        default=1,
                        help='Batch size.')

    parser.add_argument('-sequence_length',
                        type=int,
                        default=8,
                        help='Sequence length.')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # Gathers the command line arguments
    args = parse_args()

    # Adds some properties that may impact performance
    options = SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    # Creates the onnxruntime session (standard)
    session = InferenceSession(args.onnx_model_path, options)
    session.disable_fallback()

    # Creates the onnxruntime session (quantized)
    session_qnt = InferenceSession(args.onnx_model_path, options)
    session_qnt.disable_fallback()

    # Tokenizes the input text into tokens
    inputs = {
        'data': torch.randint(1, 1000, (args.batch_size, args.sequence_length))
    }
    input_onnx = {k: v.cpu().detach().numpy() for k, v in inputs.items()}

    # Performs the inference and compares the outputs
    logits = session.run(None, input_onnx)[0]
    logits_qnt = session_qnt.run(None, input_onnx)[0]

    print(f'Difference between logits: {(logits != logits_qnt).sum() / logits.shape[-1] * 100}%')
