#pragma once
#include <vector>
#include <string>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <glm/glm.hpp>

#include "niftilib/nifti2_io.h"
#include <imgui/imgui.h>


enum class Format
{
    NIFTI, UNKNOWN
};

Format getFileFormat(const char* path);

short* readVolume(const char* path, int& width, int& height, int& depth, float& scaleX, float& scaleY, float& scaleZ);
void saveVolume(const char* path, short* data, const int width, const int height, const int depth);

void readTF(const char* path, std::vector<ImVec2>& alphaPoints, std::vector<float>& colors);
void saveTF(const char* path, std::vector<ImVec2> alphaPoints, std::vector<float> colors);