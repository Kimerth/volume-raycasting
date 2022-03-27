#include "Loader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>


Format getFileFormat(const char* path)
{
    char* s = new char[strlen(path) + 1];
    memcpy(s, path, strlen(path) + 1);

    char* last_tok = s;
    for (char* tok = strchr(s, '.'); tok; last_tok = tok, tok = strchr(tok + 1, '.'));
    last_tok++;

    if (strcmp(last_tok, "nii") == 0 || strcmp(last_tok, "gz") == 0)
        return Format::NIFTI;
    else
        return Format::UNKNOWN;
}

USHORT* readNIFTI(const char* path, int& width, int& height, int& depth, float& scaleX, float& scaleY, float& scaleZ)
{
    nifti_image* nim;

    nim = nifti_image_read(path, 1);

    width = nim->nx, height = nim->ny, depth = nim->nz;

    scaleX = nim->dx; scaleY = nim->dy, scaleZ = nim->dz;

    USHORT* data;

    switch (nim->datatype)
    {
    case NIFTI_TYPE_INT16:
        data = (USHORT*)nim->data;
        break;
    default:
        throw std::exception("Unexpected data type");
    }

    return data;
}

USHORT* readVolume(const char* path, int& width, int& height, int& depth, float& scaleX, float& scaleY, float& scaleZ)
{
    switch (getFileFormat(path))
    {
    case Format::NIFTI:
        return readNIFTI(path, width, height, depth, scaleX, scaleY, scaleZ);
    default:
        return nullptr;
    }
}

float* readTF(const char* path)
{
    float* data = new float[4 * 256];

    std::ifstream file(path, std::ios::binary);
    if (!file.good())
        exit(EXIT_FAILURE);

    file.seekg(0, file.end);
    int length = file.tellg();
    file.seekg(0, file.beg);

    char* buffer = new char[length];
    file.read(buffer, length);

    char* ptr = strstr(buffer, "re=");
    for (int i = 0; ptr != NULL && i < 4 * 256; i+=4, ptr = strstr(ptr + 1, "re="))
    {
        float re, ge, be;
        float ra, ga, ba;
        sscanf_s(ptr,
            "re=%f\n"
            "ge=%f\n"
            "be=%f\n"
            "ra=%f\n"
            "ga=%f\n"
            "ba=%f\n", &re, &ge, &be, &ra, &ga, &ba);
        data[i]     = re;
        data[i + 1] = ge;
        data[i + 2] = be;
        data[i + 3] = (ra + ga + ba) / 3;
    }

    return data;
}
