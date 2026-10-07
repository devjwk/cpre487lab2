#pragma once

#include <cstring>

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"

namespace ML {
// Flatten (H, W, C) into one vector of H * W * C values
class FlattenLayer : public Layer {
   public:
    // The framework has no layer type for flatten, so LayerType::NONE is used
    FlattenLayer(const LayerParams inParams, const LayerParams outParams) : Layer(inParams, outParams, LayerType::NONE) {}

    // Feature maps are already stored as one array in (H, W, C) order, which is the order Keras flattens in,
    // so the values are copied unchanged and only the dimensions differ
    virtual void computeNaive(const LayerData& dataIn) const override {
        std::memcpy(getOutputData().raw(), dataIn.raw(), getOutputParams().byte_size());
    }

    // Only the naive implementation is required for this lab
    virtual void computeThreaded(const LayerData& dataIn) const override { computeNaive(dataIn); }
    virtual void computeTiled(const LayerData& dataIn) const override { computeNaive(dataIn); }
    virtual void computeSIMD(const LayerData& dataIn) const override { computeNaive(dataIn); }
};

}  // namespace ML
