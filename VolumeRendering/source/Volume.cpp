#include "Volume.h"
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
        GLuint textures[] = { texID, gradsID, segID };
        glDeleteTextures(3, textures);
    }

    if (volumeData)
        delete[] volumeData;
	if(segmentationData)
        delete[] segmentationData;

    {
        glGenTextures(1, &texID);
        glBindTexture(GL_TEXTURE_3D, texID);

        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

        volumeData = readVolume(path, sizeX, sizeY, sizeZ, scale.x, scale.y, scale.z);
        int maxSize = std::max({sizeX, sizeY, sizeZ});
        // TODO not right: look at affine transformation
        // https://nipy.org/nibabel/coordinate_systems.html#the-affine-matrix-as-a-transformation-between-spaces
        scale.x *= (float)sizeX / maxSize, scale.y *= (float)sizeY / maxSize, scale.z *= (float)sizeZ / maxSize;
        scale /= std::max({ scale.x, scale.y, scale.z });

        glTexImage3D(GL_TEXTURE_3D, 0, GL_LUMINANCE, sizeX, sizeY, sizeZ, 0, GL_LUMINANCE, GL_SHORT, volumeData);

        // TODO move this
        // ---
        std::memset(hist, 0, USHRT_MAX * sizeof(float));
        //for (int i = 0; i < sizeX; ++i)
        //    for (int j = 0; j < sizeY; ++j)
        //        for (int k = 0; k < sizeZ; ++k)
        //            hist[volumeData[(k * sizeX * sizeY) + (j * sizeX) + i] - SHRT_MIN]++;

        //hist[0] = 0;
        //for (int i = 0; i < USHRT_MAX; ++i)
        //    hist[i] = std::log(hist[i] + 1);
        //float max = *std::max_element(hist, hist + USHRT_MAX);
        //float min = *std::min_element(hist + 1, hist + USHRT_MAX);
        //for (int i = 1; i < USHRT_MAX; ++i)
        //    hist[i] = (hist[i] - min) / max;
        // ---

        glGenTextures(1, &segID);
        glBindTexture(GL_TEXTURE_3D, segID);

        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

        GLubyte* buffer = new GLubyte[sizeX * sizeY * sizeZ];
        memset(buffer, 255, sizeX * sizeY * sizeZ);
        glTexImage3D(GL_TEXTURE_3D, 0, GL_LUMINANCE, sizeX, sizeY, sizeZ, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, buffer);
        delete[] buffer;
    }

    {
        glGenTextures(1, &gradsID);
        glBindTexture(GL_TEXTURE_3D, gradsID);

        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

        glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB16F, sizeX, sizeY, sizeZ, 0, GL_RGB, GL_FLOAT, NULL);
    }
}

void Volume::loadSegmentation(const char* path)
{
    short* buffer = readVolume(path, sizeX, sizeY, sizeZ, scale.x, scale.y, scale.z, false);

    size_t size = sizeX * sizeY * sizeZ;
    segmentationData = new uchar[size];
    for (int i = 0; i < size; ++i)
        segmentationData[i] = (uchar)((buffer[i] + SHRT_MAX) / ((2 * SHRT_MAX) / 5));

    //for (int i = 0; i < size; ++i)
    //    if (segmentationData[i] > 0)
    //        std::cout << buffer[i] << '\t' << +segmentationData[i] << std::endl;
	
    delete[] buffer;

    applySegmentation();
}

void smoothLabel(uchar* labels, int radius, int x, int y, int z, int width, int height, int depth)
{
    int freq[7];
    std::memset(freq, 0, 7 * sizeof(int));

    for (int i = x - radius; i <= x + radius; ++i)
        for (int j = y - radius; j <= y + radius; ++j)
            for (int k = z - radius; k <= z + radius; ++k)
                if (i >= 0 && i < width && j >= 0 && j < height && k >= 0 && k < depth)
                    freq[labels[k * width * height + j * width + i]] ++;

    labels[z * width * height + y * width + x] = std::distance(freq, std::max_element(freq, freq + 7));
}

void Volume::applySmoothingLabels()
{
    if (smoothedSegmentationData != nullptr)
        delete[] smoothedSegmentationData;

    smoothedSegmentationData = new uchar[sizeX * sizeY * sizeZ];
	std::memcpy(smoothedSegmentationData, segmentationData, sizeX * sizeY * sizeZ);
	
    if (smoothingRadius > 0)
    {
        std::thread smoothingThreadObj([](Volume* v) {
            for (int x = 0; x < v->sizeX; ++x)
                for (int y = 0; y < v->sizeY; ++y)
                    for (int z = 0; z < v->sizeZ; ++z)
                        smoothLabel(v->smoothedSegmentationData, v->smoothingRadius, x, y, z, v->sizeX, v->sizeY, v->sizeZ);
        }, this);

        smoothingThreadObj.detach();
    }
}

void Volume::computeSegmentation(PytorchModel ptModel)
{
    std::thread segmentThreadObj([](Volume *v, PytorchModel ptModel) {
        v->segmentationData = ptModel.forward(v->volumeData, v->sizeX, v->sizeY, v->sizeZ);

        v->applySmoothingLabels();
		
        v->applySegmentation();
    }, this, ptModel);

    segmentThreadObj.detach();
}

void Volume::applySegmentation()
{
    size_t size = sizeX * sizeY * sizeZ;
    uchar* segmentationBuffer = new uchar[size];

    for (int i = 0; i < size; ++i)
        if (smoothedSegmentationData[i] <= 7 && labelsEnabled[smoothedSegmentationData[i]])
            segmentationBuffer[i] = 255;
        else
            segmentationBuffer[i] = 0;

    glBindTexture(GL_TEXTURE_3D, segID);
    glTexImage3D(GL_TEXTURE_3D, 0, GL_INTENSITY, sizeX, sizeY, sizeZ, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, segmentationBuffer);

    delete[] segmentationBuffer;
}

void Volume::loadTF(float data[])
{
    if (tfID == NULL)
    {
        glGenTextures(1, &tfID);
        glBindTexture(GL_TEXTURE_1D, tfID);

        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    }

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

    memset(labelsEnabled, true, 7 * sizeof(bool));
}
