# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from os import environ

# Constants available in onnxruntime
# that enables performance optimization
environ["OMP_NUM_THREADS"] = str(1)
environ["OMP_WAIT_POLICY"] = 'ACTIVE'

import argparse
import time
from contextlib import contextmanager
from typing import List

import numpy as np
import torch
from onnxruntime import (GraphOptimizationLevel, InferenceSession,
                         SessionOptions)


def parse_args():
    parser = argparse.ArgumentParser(description='Measures latency with an ONNX model.')

    parser.add_argument('--onnx_model_path',
                        type=str,
                        help='Path to the pre-trained ONNX model file.')

    parser.add_argument('--batch_size',
                        type=int,
                        default=1,
                        help='Batch size.')

    parser.add_argument('--sequence_length',
                        type=int,
                        default=8,
                        help='Sequence length.')

    args = parser.parse_args()

    return args


@contextmanager
def track_inference_time(latency: List[int]) -> None:
    # Gathers time between execution
    start = time.time()
    yield
    end = time.time()

    # Appends to list
    latency.append(end - start)


if __name__ == '__main__':
    # Gathers the command line arguments
    args = parse_args()

    # Adds some properties that may impact performance
    options = SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    # Creates the onnxruntime session
    session = InferenceSession(args.onnx_model_path, options)
    session.disable_fallback()

    # Tokenizes the input text into tokens
    inputs = {
        'data': torch.randint(1, 1000, (args.batch_size, args.sequence_length))
    }
    input_onnx = {k: v.cpu().detach().numpy() for k, v in inputs.items()}

    # Performs the inference and measures the latency
    latency = []
    for _ in range(1000):
        with track_inference_time(latency):
            session.run(None, input_onnx)

    print(f'Mean inference latency: {np.mean(latency) * 1e3} ms')
    print(f'P50 inference latency: {np.percentile(latency, 50) * 1e3} ms')
    print(f'P90 inference latency: {np.percentile(latency, 90) * 1e3} ms')
    print(f'P95 inference latency: {np.percentile(latency, 95) * 1e3} ms')
