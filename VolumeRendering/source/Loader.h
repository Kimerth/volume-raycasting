#pragma once
#include <string>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <glm/glm.hpp>

#include "niftilib/nifti2_io.h"


enum class Format
{
    NIFTI, UNKNOWN
};

Format getFileFormat(const char* path);

short* readVolume(const char* path, int& width, int& height, int& depth, float& scaleX, float& scaleY, float& scaleZ, bool normalize = true);

float* readTF(const char* path);