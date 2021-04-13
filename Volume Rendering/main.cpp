#include <iostream>
#include <fstream>
#include <string>

#include <stdio.h>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <glm/mat4x4.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Volume.h"

const int windowWidth = 512, windowHeight = 512;

Volume v;

void display()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	v.draw();

	glFlush();
}

void init()
{
	glEnable(GL_TEXTURE_3D | GL_DEPTH_TEST);

	glViewport(0, 0, windowWidth, windowHeight);
	glClearColor(0.0f, 0.0f, 0.0f, 0.5f);
	glClearDepth(1.0f);
	glewInit();

	v.load("res/backpack.nrrd");
}

int main(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
	glutInitWindowPosition(200, 200);
	glutInitWindowSize(windowWidth, windowWidth);
	glutCreateWindow("Volume Rendering");

	init();

	glutDisplayFunc(display);
	glutMainLoop();

	return 0;
}
