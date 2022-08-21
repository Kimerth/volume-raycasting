#include "Loader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>


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

float mean(short* data, int size)
{
	float sum = 0;
	for (int i = 0; i < size; i++)
		sum += data[i];
	return sum / size;
}

short* readNIFTI(const char* path, glm::ivec3& size, glm::mat4& xtoi)
{
    nifti_image* nim = nifti_image_read(path, 1);

    size.x = nim->nx, size.y = nim->ny, size.z = nim->nz;

    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            xtoi[i][j] = nim->qto_ijk.m[i][j];
    
    short* data;

    switch (nim->datatype)
    {
    case NIFTI_TYPE_INT16:
        data = (short*)nim->data;
        break;
    default:
        throw std::exception("Unexpected data type");
    }

    delete nim;

    return data;
}

void writeNIFTI(const char* path, short* data, const glm::ivec3 size)
{
    int64_t dims[] = { 3, size.x, size.y, size.z};
    nifti_image* nim = nifti_make_new_nim(dims, NIFTI_TYPE_INT16, 0);
    nim->data = data;
    nim->fname = (char*)path;
    nifti_image_write(nim);

    delete nim;
}

short* readVolume(const char* path, glm::ivec3& size, glm::mat4& itox)
{
    switch (getFileFormat(path))
    {
    case Format::NIFTI:
        try
        {
            return readNIFTI(path, size, itox);
        }
        catch(std::exception e)
        {
            return nullptr;
        }
    default:
        return nullptr;
    }
}

void saveVolume(const char* path, short* data, const glm::ivec3 size)
{
    switch (getFileFormat(path))
    {
    case Format::NIFTI:
        return writeNIFTI(path, data, size);
    default:
        break;
    }
}

void readTF(const char* path, std::vector<ImVec2>& alphaPoints, std::vector<float>& colors)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.good())
    {
        std::cout << "Error opening file " << path << std::endl;
        return;
    }

    int size;
    file >> size;

    alphaPoints.resize(size);
    colors.resize( 3 * size);

    for (int i = 0; i < size; ++i)
        file >> alphaPoints[i].x >> alphaPoints[i].y >> colors[3 * i] >> colors[3 * i + 1] >> colors[3 * i + 2];

    file.close();
}

void saveTF(const char* path, std::vector<ImVec2> alphaPoints, std::vector<float> colors)
{
    std::ofstream file(path, std::ios::binary);
    if (!file.good())
    {
        std::cout << "Error opening file " << path << std::endl;
        return;
    }

    file << alphaPoints.size() << std::endl;

    for (int i = 0; i < alphaPoints.size(); ++i)
    {
        file << alphaPoints[i].x << " " << alphaPoints[i].y << " ";
        file << colors[3 * i] << " " << colors[3 * i + 1] << " " << colors[3 * i + 2] << std::endl;
    }

    file.close();
}
