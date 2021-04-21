#pragma once
#include <GL/glew.h>
#include <GL/freeglut.h>

class Volume
{
public:
	GLuint vao;
	GLuint texID;
	GLuint backFaceID;

	Volume();
	void load(const char* path);
private:
	int sizeX, sizeY, sizeZ;

	void init();
};