#pragma once

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"

namespace ML {
// Max pooling over non-overlapping square windows (window size = stride)
class MaxPoolingLayer : public Layer {
   public:
    MaxPoolingLayer(const LayerParams inParams, const LayerParams outParams) : Layer(inParams, outParams, LayerType::MAX_POOLING) {}

    // Each output value is the maximum of one window of the same input channel
    virtual void computeNaive(const LayerData& dataIn) const override {
        // Input (H, W, C), output (P, Q, C)
        const dimVec& inDims = getInputParams().dims;
        const dimVec& outDims = getOutputParams().dims;

        const size H = inDims[0], W = inDims[1], C = inDims[2];
        const size P = outDims[0], Q = outDims[1];

        // Window size in each direction (2 x 2 for every pooling layer of our model)
        const size R = H / P, S = W / Q;

        LayerData& dataOut = getOutputData();

        for (size p = 0; p < P; p++) {
            for (size q = 0; q < Q; q++) {
                for (size c = 0; c < C; c++) {
                    fp32 maxValue = dataIn.get<fp32>(((p * R) * W + (q * S)) * C + c);

                    for (size r = 0; r < R; r++) {
                        for (size s = 0; s < S; s++) {
                            const fp32 value = dataIn.get<fp32>(((p * R + r) * W + (q * S + s)) * C + c);
                            if (value > maxValue) maxValue = value;
                        }
                    }

                    dataOut.get<fp32>((p * Q + q) * C + c) = maxValue;
                }
            }
        }
    }

    // Only the naive implementation is required for this lab
    virtual void computeThreaded(const LayerData& dataIn) const override { computeNaive(dataIn); }
    virtual void computeTiled(const LayerData& dataIn) const override { computeNaive(dataIn); }
    virtual void computeSIMD(const LayerData& dataIn) const override { computeNaive(dataIn); }
};

}  // namespace ML
