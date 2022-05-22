#pragma once

#include <torch/script.h>
#include <torch/torch.h>


typedef unsigned short ushort;
typedef unsigned char uchar;

class PytorchModel
{
public:
	PytorchModel();
	void loadModel(const char* path);
	uchar* forward(short* data, int width, int height, int depth);

	torch::Device device = torch::kCPU;
	torch::jit::script::Module model;

	std::vector<int> inputSize;
	std::vector<int> patchSize;

	bool isLoaded = false;
};