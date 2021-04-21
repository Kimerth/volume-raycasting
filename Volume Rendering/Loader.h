#pragma once
#include <string>

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

uchar* readPVM(const char* path, int& width, int& height, int& depth);
uchar* readNRRD(const char* path, int& width, int& height, int& depth);

uchar* readVolume(const char* path, int& width, int& height, int& depth);