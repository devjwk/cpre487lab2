#include "Convolutional.h"

#include <iostream>

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"

namespace ML {
// --- Begin Student Code ---

// Compute the convultion for the layer data
void ConvolutionalLayer::computeNaive(const LayerData& dataIn) const {
    // Dimensions, named as in the textbook: input (H, W, C), weights (R, S, C, M), output (P, Q, M)
    const dimVec& inDims = getInputParams().dims;
    const dimVec& outDims = getOutputParams().dims;
    const dimVec& weightDims = getWeightParams().dims;

    const size W = inDims[1], C = inDims[2];
    const size P = outDims[0], Q = outDims[1], M = outDims[2];
    const size R = weightDims[0], S = weightDims[1];

    LayerData& dataOut = getOutputData();

    // Stride 1, no padding: output (p, q) covers input rows p..p+R-1 and columns q..q+S-1
    for (size p = 0; p < P; p++) {
        for (size q = 0; q < Q; q++) {
            for (size m = 0; m < M; m++) {
                fp32 sum = biasData.get<fp32>(m);

                for (size r = 0; r < R; r++) {
                    for (size s = 0; s < S; s++) {
                        for (size c = 0; c < C; c++) {
                            sum += dataIn.get<fp32>(((p + r) * W + (q + s)) * C + c) *
                                   weightData.get<fp32>(((r * S + s) * C + c) * M + m);
                        }
                    }
                }

                // ReLU activation is part of the layer, as in the Keras Conv2D layers of the model
                dataOut.get<fp32>((p * Q + q) * M + m) = sum > 0 ? sum : 0;
            }
        }
    }
}

// Compute the convolution using threads
void ConvolutionalLayer::computeThreaded(const LayerData& dataIn) const {
    // Only the naive implementation is required for this lab
    computeNaive(dataIn);
}

// Compute the convolution using a tiled approach
void ConvolutionalLayer::computeTiled(const LayerData& dataIn) const {
    // Only the naive implementation is required for this lab
    computeNaive(dataIn);
}

// Compute the convolution using SIMD
void ConvolutionalLayer::computeSIMD(const LayerData& dataIn) const {
    // Only the naive implementation is required for this lab
    computeNaive(dataIn);
}
}  // namespace ML
