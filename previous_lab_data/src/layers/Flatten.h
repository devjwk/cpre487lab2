#pragma once

#include "Layer.h"

namespace ML {

class FlattenLayer : public Layer {
   public:
    FlattenLayer(const LayerParams inParams,
                 const LayerParams outParams)
        : Layer(inParams, outParams, LayerType::FLATTEN) {}

    virtual void computeNaive(const LayerData& dataIn) const override {
        const std::size_t count = dataIn.getParams().flat_count();

        for (std::size_t i = 0; i < count; i++) {
            getOutputData().get<fp32>(i) =
                dataIn.get<fp32>(i);
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