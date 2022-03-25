#pragma once
#include <string>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <glm/glm.hpp>

#include "IO/nifti1_io.h"

#define DDS_BLOCKSIZE (1<<20)
#define DDS_INTERLEAVE (1<<24)
#define DDS_RL (7)

#define DDS_ISINTEL (*((unsigned char *)(&DDS_INTEL)+1)==0)

enum class Format
{
    NRRD, PVM, NIFTI, UNKNOWN
};

Format getFileFormat(const char* path);

//uchar* readPVM(const char* path, int& width, int& height, int& depth, float& scaleX, float& scaleY, float& scaleZ);
//uchar* readNRRD(const char* path, int& width, int& height, int& depth, float& scaleX, float& scaleY, float& scaleZ);

USHORT* readVolume(const char* path, int& width, int& height, int& depth, float& scaleX, float& scaleY, float& scaleZ);

float* readTF(const char* path);