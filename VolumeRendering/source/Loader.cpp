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

short* readNIFTI(const char* path, int& width, int& height, int& depth, float& scaleX, float& scaleY, float& scaleZ, bool normalize)
{
    nifti_image* nim = nifti_image_read(path, 1);

    width = nim->nx, height = nim->ny, depth = nim->nz;

    scaleX = nim->dx; scaleY = nim->dy, scaleZ = nim->dz;

    short* data;

    switch (nim->datatype)
    {
    case NIFTI_TYPE_INT16:
        data = (short*)nim->data;
        break;
    default:
        throw std::exception("Unexpected data type");
    }

    // normalize
    if (normalize)
    {
        size_t size = width * height * depth;
        float avg = mean(data, size);
	
        for (size_t i = 0; i < size; i++)
            data[i] -= avg;

        short min = *std::min_element(data, data + size);
        short max = *std::max_element(data, data + size);
        float scale = (SHRT_MAX - SHRT_MIN) / (max - min);

	    for (size_t i = 0; i < size; i++)
		    data[i] = scale * data[i];
    }

    return data;
}

void writeNIFTI(const char* path, short* data, const int width, const int height, const int depth)
{
    int64_t dims[] = { 3, width, height, depth };
    nifti_image* nim = nifti_make_new_nim(dims, NIFTI_TYPE_INT16, 0);
    nim->data = data;
    nim->fname = (char*)path;
    nifti_image_write(nim);
}

short* readVolume(const char* path, int& width, int& height, int& depth, float& scaleX, float& scaleY, float& scaleZ, bool normalize)
{
    switch (getFileFormat(path))
    {
    case Format::NIFTI:
        try
        {
            return readNIFTI(path, width, height, depth, scaleX, scaleY, scaleZ, normalize);
        }
        catch(std::exception e)
        {
            return nullptr;
        }
    default:
        return nullptr;
    }
}

void saveVolume(const char* path, short* data, const int width, const int height, const int depth)
{
    switch (getFileFormat(path))
    {
    case Format::NIFTI:
        return writeNIFTI(path, data, width, height, depth);
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
