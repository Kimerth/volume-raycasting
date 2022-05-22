#pragma once
#include <limits>
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

	int smoothingRadius = 0;

	short* volumeData;
	uchar* segmentationData;
	uchar* smoothedSegmentationData;

	bool labelsEnabled[7];

	float hist[USHRT_MAX];

	bool computingSegmentation = false;
	bool smoothingSegmentation = false;

	void load(const char* path);
	void loadSegmentation(const char* path);
	void computeSegmentation(PytorchModel ptModel);
	void loadTF(float data[]);
	void applySegmentation();
	void applySmoothingLabels();
private:
	void init();
};