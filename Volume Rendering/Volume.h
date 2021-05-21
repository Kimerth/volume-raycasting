#pragma once
#include <GL/glew.h>
#include <GL/freeglut.h>

#include <glm/mat4x4.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>

class Volume
{
public:
	GLuint vao;
	GLuint texID;
	GLuint tfID;
	GLuint gradsID;
	int sizeX, sizeY, sizeZ;
	glm::vec3 scale;

	float hist[256];

	void load(const char* path);
	void loadTF(float data[]);
private:
	void init();
};