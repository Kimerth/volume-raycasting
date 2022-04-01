#pragma once
#include <GL/glew.h>
#include <GL/freeglut.h>

#include <glm/mat4x4.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "PytorchModel.h"

class Volume
{
public:
	GLuint vao;
	GLuint texID;
	GLuint tfID;
	GLuint gradsID;
	GLuint segID;
	int sizeX, sizeY, sizeZ;
	glm::vec3 scale;

	short* volumeData;
	uchar* segmentationData;

	bool labelsEnabled[7];

	float hist[1 << 16];

	void load(const char* path);
	void loadSegmentation(const char* path);
	void computeSegmentation(PytorchModel ptModel);
	void loadTF(float data[]);
	void applySegmentation();
private:
	void init();
};