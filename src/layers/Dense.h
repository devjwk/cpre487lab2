#pragma once

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"

namespace ML {
// Fully connected layer: every output is a weighted sum of all inputs plus a bias
class DenseLayer : public Layer {
   public:
    // applyRelu is false for the last dense layer, whose output goes to the softmax layer instead
    DenseLayer(const LayerParams inParams, const LayerParams outParams, const LayerParams weightParams, const LayerParams biasParams,
               const bool applyRelu = true)
        : Layer(inParams, outParams, LayerType::DENSE),
          weightParam(weightParams),
          weightData(weightParams),
          biasParam(biasParams),
          biasData(biasParams),
          applyRelu(applyRelu) {}

    // Getters
    const LayerParams& getWeightParams() const { return weightParam; }
    const LayerParams& getBiasParams() const { return biasParam; }
    const LayerData& getWeightData() const { return weightData; }
    const LayerData& getBiasData() const { return biasData; }

    // Allocate all resources needed for the layer & Load all of the required data for the layer
    virtual void allocLayer() override {
        Layer::allocLayer();
        weightData.loadData();
        biasData.loadData();
    }

    // Free all resources allocated for the layer
    virtual void freeLayer() override {
        Layer::freeLayer();
        weightData.freeData();
        biasData.freeData();
    }

    virtual void computeNaive(const LayerData& dataIn) const override {
        // Input (N), weights (N, M), output (M)
        const size N = getInputParams().dims[0];
        const size M = getOutputParams().dims[0];

        LayerData& dataOut = getOutputData();

        for (size m = 0; m < M; m++) {
            fp32 sum = biasData.get<fp32>(m);

            for (size n = 0; n < N; n++) {
                sum += dataIn.get<fp32>(n) * weightData.get<fp32>(n * M + m);
            }

            dataOut.get<fp32>(m) = (applyRelu && sum < 0) ? 0 : sum;
        }
    }

    // Only the naive implementation is required for this lab
    virtual void computeThreaded(const LayerData& dataIn) const override { computeNaive(dataIn); }
    virtual void computeTiled(const LayerData& dataIn) const override { computeNaive(dataIn); }
    virtual void computeSIMD(const LayerData& dataIn) const override { computeNaive(dataIn); }

   private:
    LayerParams weightParam;
    LayerData weightData;

    LayerParams biasParam;
    LayerData biasData;

    bool applyRelu;
};

}  // namespace ML
