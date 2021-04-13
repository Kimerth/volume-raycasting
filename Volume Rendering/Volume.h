#pragma once
#include <GL/glew.h>
#include <GL/freeglut.h>

class Volume
{
public:
	Volume();
	void load(const char* path);
	void draw();
private:
	GLuint texID;
	int sizeX, sizeY, sizeZ, sizeData;
};