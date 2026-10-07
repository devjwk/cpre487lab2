#include "Convolutional.h"

#include <iostream>

#include "../Types.h"
#include "../Utils.h"
#include "Layer.h"

namespace ML {
// --- Begin Student Code ---

// Compute the convultion for the layer data
void ConvolutionalLayer::computeNaive(const LayerData& dataIn) const {
    // TODO: Your Code Here...
    // The following line is an example of copying a single 32-bit floating point integer from the input layer data to the output layer data
    /*getOutputData().get<fp32>(0) = dataIn.get<fp32>(0);*/
    const auto& inDims = dataIn.getParams().dims;
    const auto& outDims = getOutputParams().dims;
    const auto& weightDims = getWeightParams().dims;

    const std::size_t inW = inDims[1];
    const std::size_t inC = inDims[2];

    const std::size_t outH = outDims[0];
    const std::size_t outW = outDims[1];
    const std::size_t outC = outDims[2];

    const std::size_t kH = weightDims[0];
    const std::size_t kW = weightDims[1];

    for (std::size_t oy = 0; oy < outH; oy++) {
        for (std::size_t ox = 0; ox < outW; ox++) {
            for (std::size_t oc = 0; oc < outC; oc++) {

                fp32 sum = getBiasData().get<fp32>(oc);

                for (std::size_t ky = 0; ky < kH; ky++) {
                    for (std::size_t kx = 0; kx < kW; kx++) {
                        for (std::size_t ic = 0; ic < inC; ic++) {

                            std::size_t inputIdx =
                                (((oy + ky) * inW) + (ox + kx)) * inC + ic;

                            std::size_t weightIdx =
                                (((ky * kW) + kx) * inC + ic) * outC + oc;

                            sum +=
                                dataIn.get<fp32>(inputIdx) *
                                getWeightData().get<fp32>(weightIdx);
                        }
                    }
                }

                // ReLU
                if (sum < 0) {
                    sum = 0;
                }

                std::size_t outputIdx =
                    ((oy * outW) + ox) * outC + oc;

                getOutputData().get<fp32>(outputIdx) = sum;
            }
        }
    }

}

// Compute the convolution using threads
void ConvolutionalLayer::computeThreaded(const LayerData& dataIn) const {
    // TODO: Your Code Here...
}

// Compute the convolution using a tiled approach
void ConvolutionalLayer::computeTiled(const LayerData& dataIn) const {
    // TODO: Your Code Here...
}

// Compute the convolution using SIMD
void ConvolutionalLayer::computeSIMD(const LayerData& dataIn) const {
    // TODO: Your Code Here...
}
}  // namespace ML
