﻿#include "Volume.h"
#include "Loader.h"

#include <fstream>
#include <algorithm>
#include <thread>
#include <set>


void Volume::load(const char* path)
{
    if(vao == NULL)
        init();
    else
    {
        GLuint textures[] = { texID, segID };
        glDeleteTextures(2, textures);
    }

    {
        if (volumeData)
        {
            delete[] volumeData;
            volumeData = nullptr;
        }
        if (segmentationData)
        {
            delete[] segmentationData;
            segmentationData = nullptr;
        }
        if (smoothedSegmentationData)
        {
            delete[] smoothedSegmentationData;
            smoothedSegmentationData = nullptr;
        }

        for (int i = 0; i < 7; i++)
            segments[i].reset();
    }

    {
        glGenTextures(1, &texID);
        glBindTexture(GL_TEXTURE_3D, texID);

        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

        volumeData = readVolume(path, size, xtoi);
        int maxSize = std::max({ size.x, size.y, size.z });
        sizeCorrection.x = (float)size.x / maxSize, sizeCorrection.y = (float)size.y / maxSize, sizeCorrection.z = (float)size.z / maxSize;

        glTexImage3D(GL_TEXTURE_3D, 0, GL_LUMINANCE, size.x, size.y, size.z, 0, GL_LUMINANCE, GL_SHORT, volumeData);

        glGenTextures(1, &segID);
        glBindTexture(GL_TEXTURE_3D, segID);

        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

        GLubyte* buffer = new GLubyte[size.x * size.y * size.z];
        memset(buffer, UCHAR_MAX, size.x * size.y * size.z);
        glTexImage3D(GL_TEXTURE_3D, 0, GL_LUMINANCE, size.x, size.y, size.z, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, buffer);
        delete[] buffer;
    }
}

void Volume::loadSegmentation(const char* path)
{
    glm::ivec3 seg_size;
    glm::mat4 seg_itox;
    short* buffer = readVolume(path, seg_size, seg_itox);

    if (seg_size.x != size.x || seg_size.y != size.y || seg_size.z != size.z)
    {
        delete[] buffer;
        return;
    }

    size_t sizeT = size.x * size.y * size.z;
    segmentationData = new uchar[sizeT];
    for (int i = 0; i < sizeT; ++i)
        segmentationData[i] = (uchar)((buffer[i] + SHRT_MAX) / ((2 * SHRT_MAX) / 5));

    delete[] buffer;
	
    applySmoothingLabels();
	
    calculateSegmentationInfoNumVoxels();

    applySegmentation();
}

void Volume::saveSegmentation(const char* path)
{
    size_t sizeT = size.x * size.y * size.z;

    short* buffer = new short[sizeT];
    for (int i = 0; i < sizeT; ++i)
        buffer[i] = smoothedSegmentationData[i] * ((2 * SHRT_MAX) / 5) - SHRT_MAX;

    saveVolume(path, buffer, size);

    delete[] buffer;
}

uchar smoothLabel(uchar* labels, int radius, int x, int y, int z, int width, int height, int depth)
{
    int freq[7];
    std::memset(freq, 0, 7 * sizeof(int));

    for (int i = x - radius; i <= x + radius; ++i)
        for (int j = y - radius; j <= y + radius; ++j)
            for (int k = z - radius; k <= z + radius; ++k)
                if (i >= 0 && i < width && j >= 0 && j < height && k >= 0 && k < depth)
                    freq[labels[k * width * height + j * width + i]] ++;

    return std::distance(freq, std::max_element(freq, freq + 7));
}

void Volume::applySmoothingLabels(int smoothingRadius)
{
    if (smoothingSegmentation)
        return;

    if (smoothedSegmentationData != nullptr)
        delete[] smoothedSegmentationData;

    smoothedSegmentationData = new uchar[size.x * size.y * size.z];
	std::memcpy(smoothedSegmentationData, segmentationData, size.x * size.y * size.z);
	
    if (smoothingRadius > 0)
    {
        std::thread smoothingThreadObj([](Volume* v, int smoothingRadius) {
            v->smoothingSegmentation = true;
            for (int x = 0; x < v->size.x; ++x)
                for (int y = 0; y < v->size.y; ++y)
                    for (int z = 0; z < v->size.z; ++z)
                        v->smoothedSegmentationData[z * v->size.x * v->size.y + y * v->size.x + x] =
                            smoothLabel(v->segmentationData, smoothingRadius, x, y, z, v->size.x, v->size.y, v->size.z);
            v->smoothingSegmentation = false;
        }, this, smoothingRadius);

        smoothingThreadObj.detach();
    }
}

void Volume::computeSegmentation(PytorchModel ptModel)
{
    if (computingSegmentation)
        return;

    std::thread segmentThreadObj([](Volume *v, PytorchModel ptModel) {
        v->computingSegmentation = true;

        v->segmentationData = ptModel.forward(v->volumeData, v->size.x, v->size.y, v->size.z);
		
        v->applySmoothingLabels();

        v->calculateSegmentationInfoNumVoxels();
        v->applySegmentation();

        v->computingSegmentation = false;
    }, this, ptModel);

    segmentThreadObj.detach();
}

void Volume::applySegmentation()
{
    size_t sizeT = size.x * size.y * size.z;
    uchar* segmentationBuffer = new uchar[sizeT];

    for (int i = 0; i < sizeT; ++i)
        if (smoothedSegmentationData[i] < 7 && segments[smoothedSegmentationData[i]].enabled)
            segmentationBuffer[i] = UCHAR_MAX * ((float)(smoothedSegmentationData[i] + 1) / 8);
        else
            segmentationBuffer[i] = 0;

    glBindTexture(GL_TEXTURE_3D, segID);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_INTENSITY, size.x, size.y, size.z, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, segmentationBuffer);

    delete[] segmentationBuffer;
}

void Volume::applySegmentationColors()
{
    if (segColorID == NULL)
    {
        glGenTextures(1, &segColorID);
        glBindTexture(GL_TEXTURE_1D, segColorID);
		
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    }

    float colors[3 * 8];
    //for (int i = 0; i < 3; ++i)
    //    colors[i] = 1.0f;

    for (int i = 0; i < 7; ++i)
		memcpy(colors + 3 * i, segments[i].color, 3 * sizeof(float));
	
    glBindTexture(GL_TEXTURE_1D, segColorID);
    glTexImage1D(GL_TEXTURE_1D, 0, GL_RGB32F, 7, 0, GL_RGB, GL_FLOAT, colors);
}

void Volume::loadTF(float data[])
{
    if (tfID == NULL)
    {
        glGenTextures(1, &tfID);
        glBindTexture(GL_TEXTURE_1D, tfID);

        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    }

    glBindTexture(GL_TEXTURE_1D, tfID);
    glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA32F, 256, 0, GL_RGBA, GL_FLOAT, data);
}

void Volume::init()
{
    GLfloat vertices[24] = {
        0.0, 0.0, 0.0,
        0.0, 0.0, 1.0,
        0.0, 1.0, 0.0,
        0.0, 1.0, 1.0,
        1.0, 0.0, 0.0,
        1.0, 0.0, 1.0,
        1.0, 1.0, 0.0,
        1.0, 1.0, 1.0
    };

    GLuint indices[36] = {
        // front
        1,5,7,
        7,3,1,
        //back
        0,2,6,
        6,4,0,
        //left
        0,1,3,
        3,2,0,
        //right
        7,5,4,
        4,6,7,
        //up
        2,3,7,
        7,6,2,
        //down
        1,0,4,
        4,5,1
    };

    GLuint gbo[2];

    glGenBuffers(2, gbo);
    GLuint vertexdat = gbo[0];
    GLuint veridxdat = gbo[1];
    glBindBuffer(GL_ARRAY_BUFFER, vertexdat);
    glBufferData(GL_ARRAY_BUFFER, 24 * sizeof(GLfloat), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, veridxdat);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 36 * sizeof(GLuint), indices, GL_STATIC_DRAW);

    glGenVertexArrays(1, &vao);

    glBindVertexArray(vao);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ARRAY_BUFFER, vertexdat);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (GLfloat*)NULL);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (GLfloat*)NULL);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, veridxdat);
}

void Volume::calculateSegmentationInfoNumVoxels()
{
	for(int i = 0; i < size.x * size.y * size.z; ++i)
		if(segmentationData[i] < 7)
			segments[segmentationData[i]].numVoxels++;
}
