#pragma once
#include <string>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <glm/glm.hpp>

#include "IO/nifti1_io.h"

enum class Format
{
    NIFTI, UNKNOWN
};

Format getFileFormat(const char* path);

USHORT* readVolume(const char* path, int& width, int& height, int& depth, float& scaleX, float& scaleY, float& scaleZ);

float* readTF(const char* path);