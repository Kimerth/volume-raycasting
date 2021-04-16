#include "Volume.h"
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>

Volume::Volume()
    : texID(NULL), sizeX(0), sizeY(0), sizeZ(0), sizeData(0), vao(NULL), backFaceID(NULL)
{   
}

void Volume::load(const char* path, const char* tffPath)
{
    init();

    {
        glGenTextures(1, &backFaceID);
        glBindTexture(GL_TEXTURE_2D, backFaceID);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, 800, 800, 0, GL_RGBA, GL_FLOAT, NULL);
    }

    {
        std::ifstream file(path, std::ios::binary);

        if (!file.good()) return;

        for (std::string line; std::getline(file, line) && !line.empty(); )
        {
            if (line.compare(0, 6, "sizes:") == 0)
            {
                std::istringstream iss(line.substr(6));

                iss >> sizeX >> sizeY >> sizeZ;
                sizeData = sizeX * sizeY * sizeZ;
            }
        }

        {
            GLubyte* buffer = new GLubyte[sizeData];
            file.read(reinterpret_cast<char*>(buffer), sizeData);

            glGenTextures(1, &texID);
            glBindTexture(GL_TEXTURE_3D, texID);

            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
            glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);

            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            glTexImage3D(GL_TEXTURE_3D, 0, GL_INTENSITY, sizeX, sizeY, sizeZ, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, buffer);

            delete[] buffer;
        }
        file.close();
    }
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
