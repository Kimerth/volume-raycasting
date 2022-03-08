#include "PytorchModel.h"


void PytorchModel::loadModel(const char* path)
{
    try {
        model = torch::jit::load(path, torch::kCPU);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
    }
}

uchar* PytorchModel::forward(uchar* data, int width, int height, int depth)
{
    namespace F = torch::nn::functional;

    float* dataFloat = new float[width * height * depth];
    for (int i = 0; i < width * height * depth; ++i)
        dataFloat[i] = 1 / static_cast<float>(data[i]);

    torch::Tensor dataTensor = torch::from_blob(
        dataFloat,
        { width, height, depth },
        torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(torch::kCPU)
    ).unsqueeze(0).unsqueeze(0);
    dataTensor = F::interpolate(
        dataTensor,
        F::InterpolateFuncOptions()
        .mode(torch::kNearest)
        .size(std::vector<int64_t>{ 96, 96, 96 })
    );

    dataTensor = dataTensor.squeeze();

    std::vector<torch::Tensor> patches{ dataTensor };
    for (int i = 0; i < 3; ++i)
    {
        auto oldPatchesNb = std::distance(patches.begin(), patches.end());

        for (int idx = 0; idx < oldPatchesNb; ++idx)
        {
            std::vector<torch::Tensor> patchesNew = torch::split(patches[idx], 32, i);
            patches.insert(patches.end(), patchesNew.begin(), patchesNew.end());
        }

        patches.erase(patches.begin(), patches.begin() + oldPatchesNb);
    }
    std::vector<torch::Tensor> patchData;
    for (torch::Tensor& patch : patches)
        // 1, 32, 32, 32
        patchData.push_back(patch.unsqueeze(0).unsqueeze(0));

    // [B, 1, 32, 32, 32]
    torch::Tensor nnData = torch::vstack(patchData);

    std::vector<torch::jit::IValue> nnInput{ nnData };

    // expect [B, L, 32, 32, 32] ; L = nb labels
    torch::Tensor outputTensor = model.forward(nnInput).toTensor();

    std::cout << outputTensor.sizes() << std::endl;

    // expect [L, 96, 96, 96]
    //outputTensor = outputTensor.view({-1, 96, 96, 96});
    
    patches.clear();
    {
        std::vector<torch::Tensor> patchesNew = torch::split(outputTensor, 1, 0);
        for (auto patch : patchesNew)
            patches.push_back(patch.squeeze());
    }


    for (int i = 3; i > 0; --i)
    {
        auto oldPatchesNb = std::distance(patches.begin(), patches.end());

        for (int idx = 0; idx < oldPatchesNb; idx += 3)
        {
            std::vector<torch::Tensor> seq;
            seq.insert(seq.begin(), patches.begin() + idx, patches.begin() + idx + 3);
            torch::Tensor patch = torch::cat(seq, i);
            patches.push_back(patch);
        }
        std::cout << std::endl;
        patches.erase(patches.begin(), patches.begin() + oldPatchesNb);
    }

    outputTensor = patches[0].squeeze();

    std::cout << outputTensor.sizes() << std::endl;

    // expect [L, W, H, D]
    outputTensor = F::interpolate(
        outputTensor.unsqueeze(0),
        F::InterpolateFuncOptions()
        .mode(torch::kNearest)
        .size(std::vector<int64_t>({ width, height, depth }))
    );

    std::cout << outputTensor.sizes() << std::endl;

    // TODO standardize output in Pytorch: same shape as input
    float* oneHotOutput = outputTensor.contiguous().data_ptr<float>();

    uchar* output = new uchar[width * height * depth];
    int nbLabels = outputTensor.sizes().at(0);
    for (int i = 0; i < nbLabels; ++i)
        for (int k = 0; k < width * height * depth; ++k)
            output[k] = static_cast<uchar>(oneHotOutput[i + nbLabels + k]) * i;

    // expect [W, H, D]
    return output;
}
