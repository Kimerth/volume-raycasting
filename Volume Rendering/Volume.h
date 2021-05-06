#pragma once
#include <GL/glew.h>
#include <GL/freeglut.h>

class Volume
{
public:
	GLuint vao;
	GLuint texID;
	GLuint tfID;
	GLuint gradsID;
	int sizeX, sizeY, sizeZ;

	float hist[256];

	void load(const char* path);
	void loadTF(const char* path);
private:
	void init();
};