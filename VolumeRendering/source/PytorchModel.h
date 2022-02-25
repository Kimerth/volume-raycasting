#pragma once

#include <torch/script.h>


torch::jit::script::Module loadModel(const char* path);