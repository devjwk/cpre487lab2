#pragma once

#include "Layer.h"

namespace ML {

class MaxPoolingLayer : public Layer {
   public:
    MaxPoolingLayer(const LayerParams inParams,
                    const LayerParams outParams)
        : Layer(inParams, outParams, LayerType::MAX_POOLING) {}

    virtual void computeNaive(const LayerData& dataIn) const override {
        const auto& inDims = dataIn.getParams().dims;
        const auto& outDims = getOutputParams().dims;

        const std::size_t inW = inDims[1];
        const std::size_t channels = inDims[2];

        const std::size_t outH = outDims[0];
        const std::size_t outW = outDims[1];

        // Lab 1 model uses 2x2 max pooling with stride 2
        for (std::size_t oy = 0; oy < outH; oy++) {
            for (std::size_t ox = 0; ox < outW; ox++) {
                for (std::size_t c = 0; c < channels; c++) {

                    fp32 maxVal =
                        dataIn.get<fp32>(
                            (((oy * 2) * inW) + (ox * 2)) * channels + c
                        );

                    for (std::size_t ky = 0; ky < 2; ky++) {
                        for (std::size_t kx = 0; kx < 2; kx++) {

                            std::size_t inputIdx =
                                ((((oy * 2) + ky) * inW)
                                 + ((ox * 2) + kx))
                                 * channels + c;

                            fp32 value =
                                dataIn.get<fp32>(inputIdx);

                            if (value > maxVal) {
                                maxVal = value;
                            }
                        }
                    }

                    std::size_t outputIdx =
                        ((oy * outW) + ox) * channels + c;

                    getOutputData().get<fp32>(outputIdx) = maxVal;
                }
            }
        }
    }

    virtual void computeThreaded(const LayerData& dataIn) const override {
        computeNaive(dataIn);
    }

    virtual void computeTiled(const LayerData& dataIn) const override {
        computeNaive(dataIn);
    }

    virtual void computeSIMD(const LayerData& dataIn) const override {
        computeNaive(dataIn);
    }
};

} // namespace ML