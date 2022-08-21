#include "PytorchModel.h"


PytorchModel::PytorchModel()
{
    if (torch::cuda::is_available())
    {
        std::cout << "CUDA is available" << std::endl;
        std::cout << "CUDA DEVICE COUNT: " << torch::cuda::device_count() << std::endl;
        //device = torch::kCUDA;
        //device.set_index(0);
        //std::cout << "Using CUDA device: " << device.index() << std::endl;
    }
}

void PytorchModel::loadModel(const char* path)
{
    try {
        isLoaded = false;
		
        model = torch::jit::load(path, device);

        size_t dotPos = std::string(path).find_last_of(".");
		std::string modelName = std::string(path).substr(0, dotPos);

        isLoaded = true;
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model " << std::endl;
        std::cerr << e.msg() << std::endl;
    }
}

uchar* PytorchModel::forward(short* data, int width, int height, int depth)
{
	// iW, iH, iD = input image size
    int inputSize[3];
    // pW, pH, pD = output image size
    int patchSize[3];
    memcpy(inputSize, SettingsEditor::segmentationSettings.inputSize, 3 * sizeof(int));
    memcpy(patchSize, SettingsEditor::segmentationSettings.patchSize, 3 * sizeof(int));

    std::cout << device.index() << std::endl;

    namespace F = torch::nn::functional;

    size_t size = width * height * depth;

	// [W, H, D]
    torch::Tensor dataTensor = torch::from_blob(
        data,
        { 1, 1, depth, height, width },
        torch::TensorOptions()
        .dtype(torch::kInt16)
    ).to(torch::kFloat32).to(device);

    dataTensor = (dataTensor - dataTensor.mean()) / dataTensor.std();

    std::cout << device.index() << std::endl;

    dataTensor = dataTensor.transpose(2, 4);
    
    std::cout << dataTensor.sizes() << std::endl;

	// [iW, iH, iD]
    dataTensor = F::interpolate(
        dataTensor,
        F::InterpolateFuncOptions()
        .mode(torch::kNearest)
        .size(std::vector<int64_t>{ inputSize[0], inputSize[1], inputSize[2] })
    );

    std::cout << dataTensor.sizes() << std::endl;

    dataTensor = dataTensor.squeeze();

    std::vector<torch::Tensor> patches{ dataTensor };
    for (int i = 0; i < 3; ++i)
    {
        auto oldPatchesNb = std::distance(patches.begin(), patches.end());

        for (int idx = 0; idx < oldPatchesNb; ++idx)
        {
            std::vector<torch::Tensor> patchesNew = torch::split(patches[idx], patchSize[i], i);
            patches.insert(patches.end(), patchesNew.begin(), patchesNew.end());
        }

        patches.erase(patches.begin(), patches.begin() + oldPatchesNb);
    }
    std::vector<torch::Tensor> patchData;
    for (torch::Tensor& patch : patches)
        // [1, pW, pH, pD]
        patchData.push_back(patch.unsqueeze(0));

    // [B, 1, pW, pH, pD]
    torch::Tensor nnData = torch::stack(patchData);

    std::cout << nnData.sizes() << std::endl;

    std::vector<torch::jit::IValue> nnInput{ nnData };

    // expect [B, L, pW, pH, pD] ; L = nb labels
    torch::Tensor outputTensor = model.forward(nnInput).toTensor();

    std::cout << outputTensor.sizes() << std::endl;

    patches.clear();
    {
        std::vector<torch::Tensor> patchesNew = torch::split(outputTensor, 1, 0);
        for (auto patch : patchesNew)
            patches.push_back(patch.squeeze());
    }

    for (int i = 2; i >= 0; --i)
    {
        auto oldPatchesNb = std::distance(patches.begin(), patches.end());

        for (int idx = 0; idx < oldPatchesNb; idx += inputSize[i] / patchSize[i])
        {
            std::vector<torch::Tensor> seq;
            seq.insert(seq.begin(), patches.begin() + idx, patches.begin() + idx + inputSize[i] / patchSize[i]);
            torch::Tensor patch = torch::cat(seq, i + 1);
            patches.push_back(patch);
        }
        patches.erase(patches.begin(), patches.begin() + oldPatchesNb);
    }

    outputTensor = patches[0].unsqueeze(0);

    std::cout << outputTensor.sizes() << std::endl;

    outputTensor = outputTensor.transpose(2, 4);

    // [L, W, H, D]
    outputTensor = F::interpolate(
        outputTensor,
        F::InterpolateFuncOptions()
        .mode(torch::kNearest)
        .size(std::vector<int64_t>({ depth, height, width }))
    );

    std::cout << outputTensor.sizes() << std::endl;

    int nbLabels = outputTensor.sizes().at(1);

    outputTensor = torch::sigmoid(outputTensor) > 0.5;
	
    outputTensor = torch::mul(outputTensor, torch::arange(1, nbLabels + 1).to(device).reshape({ 1, nbLabels, 1, 1, 1 }));
    std::cout << outputTensor.sizes() << std::endl;

    // [W, H, D]
    outputTensor = std::get<0>(torch::max(outputTensor, 1));
    std::cout << outputTensor.sizes() << std::endl;

    outputTensor = torch::flatten(outputTensor);
    std::cout << outputTensor.sizes() << std::endl;

	uchar* result = new uchar[size];
	std::memcpy(result, outputTensor.to(torch::kUInt8).contiguous().data_ptr<uchar>(), size * sizeof(uchar));

    return result;
}
