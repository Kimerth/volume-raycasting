#include "PytorchModel.h"


torch::jit::script::Module loadModel(const char* path)
{
    torch::jit::script::Module module;

    try {
        module = torch::jit::load(path);
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
    }

    return module;
}
