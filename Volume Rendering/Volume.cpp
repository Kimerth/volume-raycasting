﻿#include "Volume.h"
#include "Loader.h"

#include <fstream>

void Volume::load(const char* path)
{
    if(vao == NULL)
        init();
    else
    {
        GLuint textures[] = { texID, gradsID, tfID };
        glDeleteTextures(3, textures);
    }

    {
        glGenTextures(1, &texID);
        glBindTexture(GL_TEXTURE_3D, texID);

        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

        GLubyte* buffer = readVolume(path, sizeX, sizeY, sizeZ);

        glTexImage3D(GL_TEXTURE_3D, 0, GL_INTENSITY, sizeX, sizeY, sizeZ, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, buffer);

        std::memset(hist, 0, 256 * sizeof(float));
        for (int i = 0; i < sizeX; ++i)
            for (int j = 0; j < sizeY; ++j)
                for (int k = 0; k < sizeZ; ++k)
                    hist[buffer[(k * sizeX * sizeY) + (j * sizeX) + i]]++;

        delete[] buffer;
    }

    {
        std::string savPath = (std::string(path) + ".sav");
        if (std::fstream{ savPath })
            loadTF(savPath.c_str());
        else
            loadTF(nullptr);
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

void Volume::loadTF(const char* path)
{
    glGenTextures(1, &tfID);
    glBindTexture(GL_TEXTURE_1D, tfID);

    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    float* buffer;
    if(path != nullptr)
        buffer = readTF(path);
    else
    {
        buffer = new float[256];
        for (int i = 0; i < 256; ++i)
            buffer[i] = 1;
    }

    glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA32F, 256, 0, GL_RGBA, GL_FLOAT, buffer);

    delete[] buffer;
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
