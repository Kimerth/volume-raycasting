#pragma once

#include <torch/script.h>
#include <torch/torch.h>


typedef unsigned char uchar;

class PytorchModel
{
public:
	void loadModel(const char* path);
	uchar* forward(uchar* data, int width, int height, int depth);

private:
	torch::jit::script::Module model;
};