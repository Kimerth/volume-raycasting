#include <iostream>
#include <fstream>
#include <string>

#include <chrono>
#include <thread>

#include <stdio.h>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <glm/mat4x4.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Volume.h"
#include "Shader.h"

#define FRAME_DURATON 32
#define ANGLE_SPEED 0.68 // approx 40degrees/sec

const int windowWidth = 800, windowHeight = 800;

Volume v;
Shader s;
glm::mat4 projection, view, model;
float angle;

glm::vec3 eyePos(0.0f, 0.0f, 1.5f);

void render();
void setShaderValues();

void display()
{
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

	glEnable(GL_DEPTH_TEST);

	glViewport(0, 0, windowWidth, windowHeight);

	render();

	glutSwapBuffers();

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	int passed = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
	std::this_thread::sleep_for(std::chrono::microseconds(std::max(0, FRAME_DURATON * 1000 - passed)));

	int deltaTime = std::max(FRAME_DURATON * 1000, passed);

	angle += ANGLE_SPEED * deltaTime / 1e+6;
}

void setShaderValues()
{
	s.setVec2("screen", windowWidth, windowHeight);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_3D, v.texID);
	s.setInt("volumeTex", 0);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_1D, v.tfID);
	s.setInt("tf", 1);

	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_3D, v.gradsID);
	s.setInt("gradsTex", 2);
}

void render() 
{
	glClearColor(0, 0, 0, 1.0f);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	model = glm::rotate((float)angle, glm::vec3(0.0f, 1.0f, 0.0f));
	model *= glm::rotate(3.14f, glm::vec3(1.0f, 0.0f, 0.0f));
	model *= glm::translate(glm::vec3(-0.5f, -0.5f, -0.5f));

	s.setMat4("modelMatrix", model);
	s.setVec3("origin", glm::vec4(eyePos, 0) * model);

	model = view * model;
	s.setMat4("viewMatrix", model);
	s.setMat4("MVP", projection * model);

	glBindVertexArray(v.vao);
	glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, (GLuint*)NULL);
}

void init()
{
	glViewport(0, 0, windowWidth, windowHeight);
	glewInit();

	v.load("res/Bonsai2-HI.pvm");

	s.load("raycasting.vert", "raycasting.frag", v);
	s.use();
	setShaderValues();

	glEnable(GL_DEPTH_TEST);

	projection = glm::perspective(1.57f, (GLfloat)windowWidth / windowHeight, 1.0f, 100.f);
	view = glm::lookAt(eyePos,
				       glm::vec3(0.0f, 0.0f, 0.0f),
					   glm::vec3(0.0f, 1.0f, 0.0f));
}

int main(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
	glutInitWindowPosition(200, 200);
	glutInitWindowSize(windowWidth, windowWidth);
	glutCreateWindow("Volume Rendering");

	init();

	glutDisplayFunc(display);
	glutIdleFunc(glutPostRedisplay);
	glutMainLoop();

	return 0;
}
