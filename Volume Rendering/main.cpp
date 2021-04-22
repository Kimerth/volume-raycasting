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
#include "Shader.h"

const int windowWidth = 800, windowHeight = 800;

Volume v;
Shader b, r;
glm::mat4 projection, view, model;
float angle;
GLuint frameBuffer;

void render(GLenum cullFace, const Shader &s);
void setShaderValues(const Shader &s);

void display()
{
	glEnable(GL_DEPTH_TEST);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, frameBuffer);
	glViewport(0, 0, windowWidth, windowHeight);

	b.use();

	render(GL_FRONT, b);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0, windowWidth, windowHeight);
	r.use();
	setShaderValues(r);
	
	render(GL_BACK, r);

	glutSwapBuffers();

	angle += 0.01;
}

void setShaderValues(const Shader &s)
{
	s.setVec2("screen", windowWidth, windowHeight);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, v.backFaceID);
	glUniform1i(glGetUniformLocation(s.shader_programme, "tex"), 1);

	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_3D, v.texID);
	glUniform1i(glGetUniformLocation(s.shader_programme, "volumeTex"), 2);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_1D, v.tfID);
	glUniform1i(glGetUniformLocation(s.shader_programme, "tf"), 0);
}

void render(GLenum cullFace, const Shader &s) 
{
	glClearColor(0, 0, 0, 1.0f);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	model = glm::mat4(1.0f);
	model *= glm::rotate((float)angle, glm::vec3(0.0f, 1.0f, 0.0f));
	model *= glm::rotate(3.14f, glm::vec3(1.0f, 0.0f, 0.0f));
	model *= glm::translate(glm::vec3(-0.5f, -0.5f, -0.5f));

	s.setMat4("modelMatrix", projection * view * model);

	glEnable(GL_CULL_FACE);
	glCullFace(cullFace);
	glBindVertexArray(v.vao);
	glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, (GLuint*)NULL);
	glDisable(GL_CULL_FACE);
}

void init()
{
	glViewport(0, 0, windowWidth, windowHeight);
	glewInit();

	v.load("res/Bonsai2-HI.pvm");

	b.load("basic.vert", "basic.frag");
	r.load("raycasting.vert", "raycasting.frag");

	GLuint depthBuffer;
	glGenRenderbuffers(1, &depthBuffer);
	glBindRenderbuffer(GL_RENDERBUFFER, depthBuffer);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, windowWidth, windowHeight);

	glGenFramebuffers(1, &frameBuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, v.backFaceID, 0);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthBuffer);
	glEnable(GL_DEPTH_TEST);

	projection = glm::perspective(90.0f, (GLfloat)windowWidth / windowHeight, 0.1f, 400.f);
	view = glm::lookAt(glm::vec3(0.0f, 0.0f, 1.0f),
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
