#include "Volume.h"
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>

Volume::Volume()
{

}

void Volume::load(const char* path)
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

    GLubyte* buffer = new GLubyte[sizeData];
    file.read(reinterpret_cast<char*>(buffer), sizeData);

    glGenTextures(1, &texID);
    glBindTexture(GL_TEXTURE_3D, texID);

    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

    glTexImage3D(GL_TEXTURE_3D, 0, GL_INTENSITY, sizeX, sizeY, sizeZ, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, buffer);

    delete[] buffer;
    file.close();
}

void Volume::draw()
{
    glEnable(GL_TEXTURE_3D);
    glBindTexture(GL_TEXTURE_3D, texID);

    for (int R = 0; R < sizeX; ++R)
    {
        glBegin(GL_QUADS);

        glTexCoord3f(0.0f, 0.0f, ((float)R + 1.0f) / 2.0f);
        glVertex3f(-1.0f, -1.0f, R);
        glTexCoord3f(1.0f, 0.0f, ((float)R + 1.0f) / 2.0f);
        glVertex3f(1.0f, -1.0f, R);
        glTexCoord3f(1.0f, 1.0f, ((float)R + 1.0f) / 2.0f);
        glVertex3f(1.0f, 1.0f, R);
        glTexCoord3f(0.0f, 1.0f, ((float)R + 1.0f) / 2.0f);
        glVertex3f(-1.0f, 1.0f, R);

        glEnd();
    }
}
