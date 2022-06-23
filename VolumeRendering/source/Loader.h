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

short* readVolume(const char* path, glm::ivec3 &size, glm::mat4 &xtoi);
void saveVolume(const char* path, short* data, const glm::ivec3 size);

void readTF(const char* path, std::vector<ImVec2>& alphaPoints, std::vector<float>& colors);
void saveTF(const char* path, std::vector<ImVec2> alphaPoints, std::vector<float> colors);