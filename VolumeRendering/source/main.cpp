﻿#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>

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
#include "PytorchModel.h"
#include "Interface.h"

#define FRAG_SHADER_PATH "source/shaders/raycasting.frag"
#define VERT_SHADER_PATH "source/shaders/raycasting.vert"

#define PYTORCH_SEGMENTATION_MODULE_PATH "segmentation_model.pt"

Volume v;
Shader s;
PytorchModel ptModel;
glm::mat4 projection, view, model;

glm::vec3 eyePos(0.0f, 0.0f, 1.5f);

Interface ui;

float deltaTime = FRAME_DURATION;

void render();

void display()
{
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

	ui.render(deltaTime);

	render();

	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

	glutSwapBuffers();

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	int passed = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
	std::this_thread::sleep_for(std::chrono::microseconds(std::max(0, FRAME_DURATION * 1000 - passed)));

	deltaTime = std::max(FRAME_DURATION * 1000, passed) / 1000;
}

void loadShaders()
{
	s.load(VERT_SHADER_PATH, FRAG_SHADER_PATH);
	s.use();

	s.setVec2("screen", ui.windowWidth, ui.windowHeight);
	s.setMat4("xtoi", v.xtoi);
	s.setVec3("scale", v.sizeCorrection);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_3D, v.texID);
	s.setInt("volumeTex", 0);

	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_3D, v.segID);
	s.setInt("segTex", 2);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_1D, v.tfID);
	s.setInt("tf", 1);

	glActiveTexture(GL_TEXTURE3);
	glBindTexture(GL_TEXTURE_1D, v.segColorID);
	s.setInt("segColors", 3);
}

void render() 
{
	v.loadTF(ui.getTFColormap());
	v.applySegmentationColors();

	glClearColor(0, 0, 0, 1.0f);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	model = glm::rotate(ui.angleX, glm::vec3(1.0f, 0.0f, 0.0f));
	model *= glm::rotate(ui.angleY, glm::vec3(0.0f, 1.0f, 0.0f));
	model *= glm::rotate(ui.angleZ, glm::vec3(0.0f, 0.0f, 1.0f));
	model *= glm::translate(glm::vec3(-0.5f, -0.5f, -0.5f));
	model *= glm::translate(glm::vec3(-ui.translationX, -ui.translationY, -ui.translationZ));

	s.setVec3("translation", glm::vec3(ui.translationX, ui.translationY, ui.translationZ));
	
	s.setVec3("bbLow", ui.bbLow);
	s.setVec3("bbHigh", ui.bbHigh);

	s.setMat4("modelMatrix", model);
	
	ui.zoom = std::max(-0.75f, std::min(1.0f, ui.zoom));
	eyePos = glm::vec3(0.0f, 0.0f, 1.0f + ui.zoom);
	view = glm::lookAt(eyePos,
		glm::vec3(0.0f, 0.0f, 0.0f),
		glm::vec3(0.0f, 1.0f, 0.0f));	

	s.setVec3("origin", glm::vec4(eyePos, 0) * model);

	model = view * model;
	s.setMat4("viewMatrix", model);
	s.setMat4("MVP", projection * model);

	s.setFloat("intensityCorrection", ui.intensityCorrection);
	s.setFloat("exposure", ui.exposure);
	s.setFloat("gamma", ui.gamma);
	s.setInt("sampleRate", ui.sampleRate);


	if (v.vao != NULL)
	{
		glBindVertexArray(v.vao);
		glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, (GLuint*)NULL);
	}
}

void init()
{
	glViewport(0, 0, ui.windowWidth, ui.windowHeight);
	glewInit();

	glEnable(GL_DEPTH_TEST);

	projection = glm::perspective(1.57f, (GLfloat)ui.windowWidth / ui.windowHeight, 0.1f, 100.f);
	view = glm::lookAt(eyePos,
				       glm::vec3(0.0f, 0.0f, 0.0f),
					   glm::vec3(0.0f, 1.0f, 0.0f));

	ptModel.loadModel(PYTORCH_SEGMENTATION_MODULE_PATH);

	ui.initialize();

	ui.canLoadVolumeFunc([&] { return !v.computingSegmentation && !v.smoothingSegmentation; });
	ui.segmentationAvailableFunc([&] { return v.segmentationData != nullptr; });
	ui.canLoadSegmentationFunc([&] { return v.volumeData != nullptr && !v.computingSegmentation && !v.smoothingSegmentation; });
	ui.canComputeSegmentationFunc([&] { return ptModel.isLoaded && v.volumeData != nullptr && !v.computingSegmentation && !v.smoothingSegmentation; });
	ui.canSmoothSegmentationFunc([&] { return !v.computingSegmentation && !v.smoothingSegmentation && v.segmentationData != nullptr; });

	ui.computeSegmentationFunc([&] { v.computeSegmentation(ptModel); });
	ui.smoothLabelsFunc([&](int radius) { v.applySmoothingLabels(radius); });
	ui.loadShadersFunc([&] { loadShaders(); });

	ui.loadVolumeFunc([&](const char* path) { v.load(path); });
	ui.loadSegmentationFunc([&](const char* path) { v.loadSegmentation(path); });
	ui.saveSegmentationFunc([&](const char* path) { v.saveSegmentation(path); });
	ui.applySegmentationFunc([&]() { v.applySegmentation(); });
	ui.getSegmentInfoFunc([&]() { return v.segments; });

	ui.getHistogramFunc([&](int nbBins) {
		float* hist = new float[nbBins];
		memset(hist, 0, nbBins * sizeof(float));

		if (v.size.x == 0 || v.size.y == 0 || v.size.z == 0)
			return hist;

		size_t size = v.size.x * v.size.y * v.size.z;

		short* data = new short[size];

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_3D, v.texID);

		glGetTexImage(GL_TEXTURE_3D, 0, GL_LUMINANCE, GL_SHORT, data);

		short maxI = *std::max_element(data, data + size);
		short minI = *std::min_element(data, data + size);

		if (maxI == minI || maxI == 0)
			return hist;

		int bin_size = (maxI - minI) / nbBins;

		for (int i = 0; i < size; ++i)
			hist[(data[i] - minI) / bin_size] += ((float)data[i] + minI) / maxI;

		//for (int i = 0; i < nbBins; ++i)
		//	hist[i] = std::log(hist[i] + 1);
		
		float max = *std::max_element(hist, hist + nbBins);
		for (int i = 0; i < nbBins; ++i)
			hist[i] = hist[i] / max;

		delete[] data;

		return hist;
	});
}

void reshape(int w, int h)
{
	ui.windowWidth = w, ui.windowHeight = h;

	glViewport(0, 0, ui.windowWidth, ui.windowHeight);
	projection = glm::perspective(1.57f, (GLfloat)ui.windowWidth / ui.windowHeight, 0.1f, 100.f);

	s.setVec2("screen", ui.windowWidth, ui.windowHeight);

	ImGui_ImplGLUT_ReshapeFunc(w, h);
}

void keyboard_wrapper(unsigned char key, int x, int y) { ui.keyboard(key, x, y); };
void specialInput_wrapper(int key, int x, int y) { ui.specialInput(key, x, y); };
void mouse_wrapper(int button, int state, int x, int y) { ui.mouse(button, state, x, y); };
void mouseWheel_wrapper(int button, int dir, int x, int y) { ui.mouseWheel(button, dir, x, y); };
void motion_wrapper(int x, int y) { ui.motion(x, y); };

int main(int argc, char** argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
	glutInitWindowPosition(200, 200);
	glutInitWindowSize(ui.windowWidth, ui.windowHeight);
	glutCreateWindow("Volume Rendering");

	glutDisplayFunc(display);
	glutIdleFunc(glutPostRedisplay);

	ImGui::CreateContext();

	ImGui_ImplGLUT_Init();
	ImGui_ImplGLUT_InstallFuncs();
	ImGui_ImplOpenGL3_Init();

	glutReshapeFunc(reshape);
	glutKeyboardFunc(keyboard_wrapper);
	glutSpecialFunc(specialInput_wrapper);
	glutMouseFunc(mouse_wrapper);
	glutMouseWheelFunc(mouseWheel_wrapper);
	glutMotionFunc(motion_wrapper);

	init();

	glutMainLoop();

	ImGui::SaveIniSettingsToDisk(ImGui::GetIO().IniFilename);

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGLUT_Shutdown();
	ImGui::DestroyContext();

	return 0;
}
