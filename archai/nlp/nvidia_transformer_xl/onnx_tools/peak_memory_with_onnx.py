# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
from os import environ
from typing import Dict

import numpy as np
import torch
from memory_profiler import memory_usage
from onnxruntime import (GraphOptimizationLevel, InferenceSession,
                         SessionOptions)

# Constants available in onnxruntime
# that enables performance optimization
environ["OMP_NUM_THREADS"] = str(1)
environ["OMP_WAIT_POLICY"] = 'ACTIVE'


def parse_args():
    parser = argparse.ArgumentParser(description='Measures peak memory with an ONNX model.')

    parser.add_argument('onnx_model_path',
                        type=str,
                        help='Path to the pre-trained ONNX model file.')

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


def track_peak_memory(session: InferenceSession, inputs: Dict[str, np.array]) -> None:
    return session.run(None, inputs)


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

    # Performs the inference and measures the peak memory
    peak_memory = []
    for _ in range(100):
        rss = memory_usage(proc=(track_peak_memory, (session, input_onnx)),
                           max_usage=True,
                           backend='psutil',
                           include_children=False,
                           multiprocess=False)
        peak_memory.append(rss)

    print(f'Inference peak memory: {np.mean(peak_memory)} MB')
