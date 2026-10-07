#include <iostream>
#include <sstream>
#include <vector>

#include "Config.h"
#include "Model.h"
#include "Types.h"
#include "Utils.h"
#include "layers/Convolutional.h"
#include "layers/Dense.h"
#include "layers/Layer.h"
#include "layers/MaxPooling.h"
#include "layers/Softmax.h"
#include "layers/Flatten.h"
#include "layers/Softmax.h"


#ifdef ZEDBOARD
#include <file_transfer/file_transfer.h>
#endif

namespace ML {

// Build our ML toy model
Model buildToyModel(const Path modelPath) {
    Model model;
    logInfo("--- Building Toy Model ---");

    // --- Conv 1: L1 ---
    // Input shape: 64x64x3
    // Output shape: 60x60x32

    // You can pick how you want to implement your layers, both are allowed:

    // LayerParams conv1_inDataParam(sizeof(fp32), {64, 64, 3});
    // LayerParams conv1_outDataParam(sizeof(fp32), {60, 60, 32});
    // LayerParams conv1_weightParam(sizeof(fp32), {5, 5, 3, 32}, modelPath / "conv1_weights.bin");
    // LayerParams conv1_biasParam(sizeof(fp32), {32}, modelPath / "conv1_biases.bin");
    // auto conv1 = new ConvolutionalLayer(conv1_inDataParam, conv1_outDataParam, conv1_weightParam, conv1_biasParam);

    model.addLayer<ConvolutionalLayer>(
        LayerParams{sizeof(fp32), {64, 64, 3}},                                    // Input Data
        LayerParams{sizeof(fp32), {60, 60, 32}},                                   // Output Data
        LayerParams{sizeof(fp32), {5, 5, 3, 32}, modelPath / "conv1_weights.bin"}, // Weights
        LayerParams{sizeof(fp32), {32}, modelPath / "conv1_biases.bin"}            // Bias
    );

    // --- Conv 2: L2 ---
    // Input shape: 60x60x32
    // Output shape: 56x56x32
    model.addLayer<ConvolutionalLayer>(
        LayerParams{sizeof(fp32), {60, 60, 32}},                                    // Input Data
        LayerParams{sizeof(fp32), {56, 56, 32}},                                   // Output Data
        LayerParams{sizeof(fp32), {5, 5, 32, 32}, modelPath / "conv2_weights.bin"}, // Weights
        LayerParams{sizeof(fp32), {32}, modelPath / "conv2_biases.bin"}            // Bias
    );

    // --- MPL 1: L3 ---
    // Input shape: 56x56x32
    // Output shape: 28x28x32
    model.addLayer<MaxPoolingLayer>(
        LayerParams{sizeof(fp32),{56,56,32}},
        LayerParams{sizeof(fp32),{28,28,32}}
    );

    // --- Conv 3: L4 ---
    // Input shape: 28x28x32
    // Output shape: 26x26x64
    model.addLayer<ConvolutionalLayer>(
        LayerParams{sizeof(fp32), {28, 28, 32}},                                    // Input Data
        LayerParams{sizeof(fp32), {26, 26, 64}},                                   // Output Data
        LayerParams{sizeof(fp32), {3, 3, 32, 64}, modelPath / "conv3_weights.bin"}, // Weights
        LayerParams{sizeof(fp32), {64}, modelPath / "conv3_biases.bin"}            // Bias
    );

    // --- Conv 4: L5 ---
    // Input shape: 26x26x64
    // Output shape: 24x24x64
     model.addLayer<ConvolutionalLayer>(
        LayerParams{sizeof(fp32), {26, 26, 64}},                                    // Input Data
        LayerParams{sizeof(fp32), {24, 24, 64}},                                   // Output Data
        LayerParams{sizeof(fp32), {3, 3, 64, 64}, modelPath / "conv4_weights.bin"}, // Weights
        LayerParams{sizeof(fp32), {64}, modelPath / "conv4_biases.bin"}            // Bias
    );


    // --- MPL 2: L6 ---
    // Input shape: 24x24x64
    // Output shape: 12x12x64
    model.addLayer<MaxPoolingLayer>(
        LayerParams{sizeof(fp32),{24,24,64}},
        LayerParams{sizeof(fp32),{12,12,64}}
    );


    // --- Conv 5: L7 ---
    // Input shape: 12x12x64
    // Output shape: 10x10x64
    model.addLayer<ConvolutionalLayer>(
        LayerParams{sizeof(fp32), {12, 12, 64}},                                    // Input Data
        LayerParams{sizeof(fp32), {10, 10, 64}},                                   // Output Data
        LayerParams{sizeof(fp32), {3, 3, 64, 64}, modelPath / "conv5_weights.bin"}, // Weights
        LayerParams{sizeof(fp32), {64}, modelPath / "conv5_biases.bin"}            // Bias
    );

    // --- Conv 6: L8 ---
    // Input shape: 10x10x64
    // Output shape: 8x8x128
    model.addLayer<ConvolutionalLayer>(
        LayerParams{sizeof(fp32), {10, 10, 64}},                                    // Input Data
        LayerParams{sizeof(fp32), {8, 8, 128}},                                   // Output Data
        LayerParams{sizeof(fp32), {3, 3, 64, 128}, modelPath / "conv6_weights.bin"}, // Weights
        LayerParams{sizeof(fp32), {128}, modelPath / "conv6_biases.bin"}            // Bias
    );

    // --- MPL 3: L9 ---
    // Input shape: 8x8x128
    // Output shape: 4x4x128
    model.addLayer<MaxPoolingLayer>(
        LayerParams{sizeof(fp32),{8,8,128}},
        LayerParams{sizeof(fp32),{4,4,128}}
    );

    // --- Flatten 1: L10 ---
    // Input shape: 4x4x128
    // Output shape: 2048
    model.addLayer<FlattenLayer>(
        LayerParams{sizeof(fp32),{4,4,128}},
        LayerParams{sizeof(fp32),{2048}}
    );

    // --- Dense 1: L11 ---
    // Input shape: 2048
    // Output shape: 256
    model.addLayer<DenseLayer>(
        LayerParams{sizeof(fp32), {2048}},
        LayerParams{sizeof(fp32), {256}},
        LayerParams{sizeof(fp32), {2048, 256},modelPath / "dense1_weights.bin"},
        LayerParams{sizeof(fp32), {256},modelPath / "dense1_biases.bin"},

        true
    );

    //--- Dense 2: L12 ---
    // Input shape: 256
    // Output shape: 200
    model.addLayer<DenseLayer>(
    LayerParams{sizeof(fp32), {256}},
    LayerParams{sizeof(fp32), {200}},
    LayerParams{sizeof(fp32), {256, 200},
                modelPath / "dense2_weights.bin"},
    LayerParams{sizeof(fp32), {200},
                modelPath / "dense2_biases.bin"},
    false
    );

    // --- Softmax 1: L13 ---
    // Input shape: 200
    // Output shape: 200
    model.addLayer<SoftmaxLayer>(
    LayerParams{sizeof(fp32), {200}},
    LayerParams{sizeof(fp32), {200}}
    );

    return model;
}

void runBasicTest(const Model& model, const Path& basePath) {
    logInfo("--- Running Basic Test ---");

    // Load an image
    LayerData img = {{sizeof(fp32), {64, 64, 3}, "./data/image_0.bin"}};
    img.loadData();

    // Compare images
    std::cout << "Comparing image 0 to itself (max error): " << img.compare<fp32>(img) << std::endl
              << "Comparing image 0 to itself (T/F within epsilon " << ML::Config::EPSILON << "): " << std::boolalpha
              << img.compareWithin<fp32>(img, ML::Config::EPSILON) << std::endl;

    // Test again with a modified copy
    std::cout << "\nChange a value by 0.1 and compare again" << std::endl;
    
    LayerData imgCopy = img;
    imgCopy.get<fp32>(0) += 0.1;

    // Compare images
    img.compareWithinPrint<fp32>(imgCopy);

    // Test again with a modified copy
    log("Change a value by 0.1 and compare again...");
    imgCopy.get<fp32>(0) += 0.1;

    // Compare Images
    img.compareWithinPrint<fp32>(imgCopy);
}

void runLayerTest(const std::size_t layerNum, const Model& model, const Path& basePath) {
    // Load an image
    logInfo(std::string("--- Running Layer Test ") + std::to_string(layerNum) + "---");

    // Construct a LayerData object from a LayerParams one
    // LayerData img(model[layerNum].getInputParams(), test_image_files[layerNum].first);
    dimVec inDims = {2048};
    LayerData img({sizeof(fp32), inDims, basePath /"image_0_data" /"layer_9_output.bin"});
    img.loadData();

    Timer timer("Layer Inference");

    // Run inference on the model
    timer.start();
    const LayerData& output = model.inferenceLayer(img, layerNum, Layer::InfType::NAIVE);
    timer.stop();

    // Compare the output
    // Construct a LayerData object from a LayerParams one
    LayerData expected(output.getParams(), basePath / "image_0_data" / "layer_10_output.bin");
    expected.loadData();
    output.compareWithinPrint<fp32>(expected);
}

void runInferenceTest(const Model& model, const Path& basePath) {
    // Load an image
    logInfo("--- Running Inference Test ---");

    // Construct a LayerData object from a LayerParams one
    LayerData img(model[0].getInputParams(), basePath / "image_0.bin");
    img.loadData();

    Timer timer("Full Inference");

    // Run inference on the model
    timer.start();
    const LayerData& output = model.inference(img, Layer::InfType::NAIVE);
    timer.stop();

    // Compare the output
    // Construct a LayerData object from a LayerParams one
    LayerData expected(model.getOutputLayer().getOutputParams(), basePath / "image_0_data" / "layer_11_output.bin");
    expected.loadData();
    output.compareWithinPrint<fp32>(expected);
}

// get an enum to assign to a layer
std::string layerName(Layer:: LayerType t){
    switch(t){
        case Layer::LayerType::CONVOLUTIONAL:   return "Convolutional";
        case Layer::LayerType::DENSE:           return "Dense";
        case Layer::LayerType::SOFTMAX:         return "SoftMax";
        case Layer::LayerType::MAX_POOLING:     return "MaxPooling";
        case Layer::LayerType::FLATTEN:         return "Flatten";
        default:                                return "Error";
    }

}

// aggregate the timings for each indivual layers
void timingTest(const Model& model, const Path& basePath){

    logInfo("Starting Inference Tests");

    LayerData img(model[0].getInputParams(), basePath / "image_0.bin");
    img.loadData();

    const std::size_t numLayers = model.getNumLayers();
    std::vector<fp32> layerTimes(numLayers);

    const LayerData* currentLayer = &img;
    for(std::size_t i = 0; i < numLayers; i++){
        Timer timer("Layer_" + std::to_string(i) + "_" + layerName(model[i].getLType()));
        
        timer.start();
        const LayerData& out = model.inferenceLayer(*currentLayer, i, Layer::InfType::NAIVE);
        timer.stop();

        layerTimes[i] = timer.milliseconds;
        currentLayer = &out;
    }

    fp32 total = 0;
    for(auto ms : layerTimes) total += ms;

    logInfo("Layer Timing Breakdown");

    for(std::size_t i = 0; i < numLayers; i++){
        float percent;
        if(total > 0){
            percent = 100.0 * layerTimes[i] / total;
        }else{
            percent = 0;
        }
        logInfo(layerName(model[i].getLType()) + " Layer (" + std::to_string(i) + std::to_string(layerTimes[i]) + " ms) " + std::to_string(percent) + "% of total time");

    }
    logInfo("Total inference time: " + std::to_string(total) + "ms");

    LayerData expected(model.getOutputLayer().getOutputParams(), basePath / "image_0_data" / "layer_11_output.bin");
    expected.loadData();
    currentLayer->compareWithinPrint<fp32>(expected);

    logInfo("Completed Layer timings");


}


void runTests() {
    // Base input data path (determined from current directory of where you are running the command)
    Path basePath("data");  // May need to be altered for zedboards loading from SD Cards

    // Build the model and allocate the buffers
    Model model = buildToyModel(basePath / "model");
    model.allocLayers();

    // Run some framework tests as an example of loading data
    runBasicTest(model, basePath);

    // Run a layer inference test
    runLayerTest(10, model, basePath);

    // Run an end-to-end inference test
    runInferenceTest(model, basePath);

    // call our timing function
    timingTest(model, basePath);





    // Clean up
    model.freeLayers();
    std::cout << "\n\n----- ML::runTests() COMPLETE -----\n";
}

} // namespace ML

#ifdef ZEDBOARD
extern "C"
int main() {
    try {
        static FATFS fatfs;
        if (f_mount(&fatfs, "/", 1) != FR_OK) {
            throw std::runtime_error("Failed to mount SD card. Is it plugged in?");
        }
        ML::runTests();
    } catch (const std::exception& e) {
        std::cerr << "\n\n----- EXCEPTION THROWN -----\n" << e.what() << '\n';
    }
    std::cout << "\n\n----- STARTING FILE TRANSFER SERVER -----\n";
    FileServer::start_file_transfer_server();
}
#else
int main() {
    ML::runTests();
}
#endif