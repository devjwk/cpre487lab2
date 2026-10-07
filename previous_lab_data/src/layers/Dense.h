#pragma once

#include "Layer.h"

namespace ML {

class DenseLayer : public Layer {
   public:
    DenseLayer(const LayerParams inParams,
               const LayerParams outParams,
               const LayerParams weightParams,
               const LayerParams biasParams,
               bool useReLU = true)
        : Layer(inParams, outParams, LayerType::DENSE),
          weightParam(weightParams),
          weightData(weightParams),
          biasParam(biasParams),
          biasData(biasParams),
          useReLU(useReLU) {}

    const LayerParams& getWeightParams() const { return weightParam; }
    const LayerParams& getBiasParams() const { return biasParam; }

    const LayerData& getWeightData() const { return weightData; }
    const LayerData& getBiasData() const { return biasData; }

    virtual void allocLayer() override {
        Layer::allocLayer();
        weightData.loadData();
        biasData.loadData();
    }

    virtual void freeLayer() override {
        Layer::freeLayer();
        weightData.freeData();
        biasData.freeData();
    }

    virtual void computeNaive(const LayerData& dataIn) const override {
        const std::size_t inputSize =
            dataIn.getParams().flat_count();

        const std::size_t outputSize =
            getOutputParams().flat_count();

        for (std::size_t out = 0; out < outputSize; out++) {

            fp32 sum = getBiasData().get<fp32>(out);

            for (std::size_t in = 0; in < inputSize; in++) {

                std::size_t weightIdx =
                    in * outputSize + out;

                sum +=
                    dataIn.get<fp32>(in) *
                    getWeightData().get<fp32>(weightIdx);
            }

            if (useReLU && sum < 0) {
                sum = 0;
            }

            getOutputData().get<fp32>(out) = sum;
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

   private:
    LayerParams weightParam;
    LayerData weightData;

    LayerParams biasParam;
    LayerData biasData;

    bool useReLU;
};

} // namespace ML