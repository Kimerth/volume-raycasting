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
        model = torch::jit::load(path, device);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model " << std::endl;
        std::cerr << e.msg() << std::endl;
    }
}

uchar* PytorchModel::forward(short* data, int width, int height, int depth)
{
    std::cout << device.index() << std::endl;

    namespace F = torch::nn::functional;

    size_t size = width * height * depth;

    float* dataFloat = new float[size];
    for (int i = 0; i < size; ++i)
        dataFloat[i] = data[i];

    torch::Tensor dataTensor = torch::from_blob(
        dataFloat,
        { 1, 1, depth, height, width },
        torch::TensorOptions()
        .dtype(torch::kFloat32)
    ).to(device);

    std::cout << device.index() << std::endl;

    dataTensor = dataTensor.transpose(2, 4);
    
    std::cout << dataTensor.sizes() << std::endl;

    dataTensor = F::interpolate(
        dataTensor,
        F::InterpolateFuncOptions()
        .mode(torch::kNearest)
        .size(std::vector<int64_t>{ 96, 96, 96 })
    );

    //dataTensor = dataTensor.transpose(2, 4);

    std::cout << dataTensor.sizes() << std::endl;

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
        patchData.push_back(patch.unsqueeze(0));

    // [B, 1, 32, 32, 32]
    torch::Tensor nnData = torch::stack(patchData);

    std::cout << nnData.sizes() << std::endl;

    std::vector<torch::jit::IValue> nnInput{ nnData };

    // expect [B, L, 32, 32, 32] ; L = nb labels
    torch::Tensor outputTensor = model.forward(nnInput).toTensor();

    std::cout << outputTensor.sizes() << std::endl;

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
        patches.erase(patches.begin(), patches.begin() + oldPatchesNb);
    }

    outputTensor = patches[0].unsqueeze(0);

    std::cout << outputTensor.sizes() << std::endl;

    outputTensor = outputTensor.transpose(2, 4);

    // expect [L, W, H, D]
    outputTensor = F::interpolate(
        outputTensor,
        F::InterpolateFuncOptions()
        .mode(torch::kNearest)
        .size(std::vector<int64_t>({ depth, height, width }))
    );

    std::cout << outputTensor.sizes() << std::endl;

    //outputTensor = outputTensor.transpose(2, 4);

    int nbLabels = outputTensor.sizes().at(1);

    outputTensor = torch::flatten(outputTensor);
    std::cout << outputTensor.sizes() << std::endl;

    float* oneHotOutput = outputTensor.contiguous().data_ptr<float>();

    uchar* output = new uchar[size];
    memset(output, 0, size * sizeof(uchar));
    for (int i = 0; i < nbLabels; ++i)
        for (int k = 0; k < size; ++k)
        {
            float val = oneHotOutput[i * size + k];
            val = 1 / (1 + exp(-val));
            if(val > 0.5)
                output[k] = i + 1;
        }

    // expect [W, H, D]
    return output;
}
