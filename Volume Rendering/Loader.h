#pragma once
#include <string>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <glm/glm.hpp>

#define DDS_BLOCKSIZE (1<<20)
#define DDS_INTERLEAVE (1<<24)
#define DDS_RL (7)

#define DDS_ISINTEL (*((unsigned char *)(&DDS_INTEL)+1)==0)

typedef unsigned char uchar;
typedef unsigned int uint;

enum class Format
{
    NRRD, PVM, UNKNOWN
};

Format getFileFormat(const char* path);

uchar* readPVM(const char* path, int& width, int& height, int& depth, float& scaleX, float& scaleY, float& scaleZ);
uchar* readNRRD(const char* path, int& width, int& height, int& depth, float& scaleX, float& scaleY, float& scaleZ);

uchar* readVolume(const char* path, int& width, int& height, int& depth, float& scaleX, float& scaleY, float& scaleZ);

float* readTF(const char* path);