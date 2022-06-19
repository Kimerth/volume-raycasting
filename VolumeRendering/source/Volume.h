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
	struct SegmentInfo
	{
		bool enabled;
		float color[3];
		size_t numVoxels;

		SegmentInfo()
		{
			enabled = true;
			for(int i = 0; i < 3; ++i)
				color[i] = 1.0f;
			numVoxels = 0;
		}
	};
	
	GLuint vao;
	GLuint texID;
	GLuint tfID;
	GLuint gradsID;
	GLuint segID;
	GLuint segColorID;
	int sizeX, sizeY, sizeZ;
	glm::vec3 scale;

	short* volumeData;
	uchar* segmentationData;
	uchar* smoothedSegmentationData;

	SegmentInfo segments[7];

	bool computingSegmentation = false;
	bool smoothingSegmentation = false;

	void load(const char* path);
	void loadTF(float data[]);
	
	void loadSegmentation(const char* path);
	void computeSegmentation(PytorchModel ptModel);
	void calcumateSegmentationInfoNumVoxels();
	
	void applySegmentation();
	void applySegmentationColors();
	void applySmoothingLabels(int smoothingRadius = 0);
private:
	void init();
};