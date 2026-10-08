#pragma once

#include <cmath>

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"

namespace ML {
// Softmax activation: Y_i = exp(g_i) / sum over l of exp(g_l)
class SoftmaxLayer : public Layer {
   public:
    SoftmaxLayer(const LayerParams inParams, const LayerParams outParams) : Layer(inParams, outParams, LayerType::SOFTMAX) {}

    virtual void computeNaive(const LayerData& dataIn) const override {
        const size N = getInputParams().flat_count();

        LayerData& dataOut = getOutputData();

        // Subtracting the largest input from every input does not change the result (the factor exp(-max) cancels
        // in the fraction) but keeps exp() from overflowing for large inputs
        fp32 maxValue = dataIn.get<fp32>(0);
        for (size i = 1; i < N; i++) {
            if (dataIn.get<fp32>(i) > maxValue) maxValue = dataIn.get<fp32>(i);
        }

        fp32 sum = 0;
        for (size i = 0; i < N; i++) {
            const fp32 e = std::exp(dataIn.get<fp32>(i) - maxValue);
            dataOut.get<fp32>(i) = e;
            sum += e;
        }

        for (size i = 0; i < N; i++) {
            dataOut.get<fp32>(i) /= sum;
        }
    }

    // Only the naive implementation is required for this lab
    virtual void computeThreaded(const LayerData& dataIn) const override { computeNaive(dataIn); }
    virtual void computeTiled(const LayerData& dataIn) const override { computeNaive(dataIn); }
    virtual void computeSIMD(const LayerData& dataIn) const override { computeNaive(dataIn); }
};

}  // namespace ML
