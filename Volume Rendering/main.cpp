#include <iostream>
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

#include <imgui/imgui.h>
#include <imgui/backends/imgui_impl_glut.h>
#include <imgui/backends/imgui_impl_opengl3.h>

#include <imgui/FD/ImGuiFileDialog.h>

#include "Volume.h"
#include "Shader.h"
#include "transfer_function_widget.h"

#define FRAME_DURATON 32
#define ANGLE_SPEED 0.68 // approx 40degrees/sec

int windowWidth = 800, windowHeight = 800;

Volume v;
Shader s;
glm::mat4 projection, view, model;
float angleY, angleX = 3.14f;

bool autoRotate = true;

glm::vec3 eyePos(0.0f, 0.0f, 1.5f);

float zoom = 0.5f;

TransferFunctionWidget tfWidget;

static bool show_volume_window = true;
static bool show_tf_window = true;

void displayUI();
void render();

void display()
{
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGLUT_NewFrame();

	displayUI();
	ImGui::Render();
	ImGuiIO& io = ImGui::GetIO();

	render();

	ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

	glutSwapBuffers();

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	int passed = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
	std::this_thread::sleep_for(std::chrono::microseconds(std::max(0, FRAME_DURATON * 1000 - passed)));

	int deltaTime = std::max(FRAME_DURATON * 1000, passed);

	if(autoRotate)
		angleY += ANGLE_SPEED * deltaTime / 1e+6;
}

void loadShaders()
{
	s.load("raycasting.vert", "raycasting.frag", v);
	s.use();

	s.setVec2("screen", windowWidth, windowHeight);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_3D, v.texID);
	s.setInt("volumeTex", 0);

	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_3D, v.gradsID);
	s.setInt("gradsTex", 2);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_1D, v.tfID);
	s.setInt("tf", 1);
}

void displayUI()
{
	ImGui::Begin("Main", NULL, ImGuiWindowFlags_MenuBar);

	if (ImGui::BeginMenuBar())
	{
		if (ImGui::MenuItem("Show Volume Window"))
			show_volume_window = true;

		if (ImGui::MenuItem("Show TF Window"))
			show_tf_window = true;
		ImGui::EndMenuBar();
	}

	if (ImGui::Button("Reload Shaders")) 
	{
		loadShaders();
	}

	if (ImGui::CollapsingHeader("Info"))
	{
		ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
	}

	ImGui::End();

	if (show_volume_window)
	{
		ImGui::Begin("Volume", &show_volume_window, ImGuiWindowFlags_MenuBar);
		if (ImGui::BeginMenuBar())
		{
			if (ImGui::MenuItem("Open.."))
				ImGuiFileDialog::Instance()->OpenDialog("ChooseVolumeOpen", "Choose Volume", ".pvm,.nrrd", ".");

			ImGui::EndMenuBar();
		}

		if (ImGuiFileDialog::Instance()->Display("ChooseVolumeOpen"))
		{
			if (ImGuiFileDialog::Instance()->IsOk())
			{
				std::string filePathName = ImGuiFileDialog::Instance()->GetFilePathName();
				v.load(filePathName.c_str());

				loadShaders();
			}

			ImGuiFileDialog::Instance()->Close();
		}

		ImGui::PlotHistogram("Histogram", v.hist, 256, 0, NULL, 0.0f, *std::max_element(v.hist, v.hist + 256), ImVec2(0, 100.0f), 1);

		ImGui::End();
	}

	if (show_tf_window)
	{
		ImGui::Begin("Transfer Function", &show_volume_window, ImGuiWindowFlags_MenuBar);
		if (ImGui::BeginMenuBar())
		{
			if (ImGui::BeginMenu("File"))
			{
				if (ImGui::MenuItem("Open.."))
					ImGuiFileDialog::Instance()->OpenDialog("ChoseTFOpen", "Choose TF", ".sav,.txt", ".");
				if (ImGui::MenuItem("Save"))
					// TODO: save
					ImGuiFileDialog::Instance()->OpenDialog("ChoseTFSave", "Choose TF", ".sav,.txt", ".");
				ImGui::EndMenu();
			}
			ImGui::EndMenuBar();
		}

		tfWidget.draw_ui();

		if (ImGuiFileDialog::Instance()->Display("ChoseTFOpen"))
		{
			if (ImGuiFileDialog::Instance()->IsOk())
			{
				std::string filePathName = ImGuiFileDialog::Instance()->GetFilePathName();
				v.loadTF(filePathName.c_str());
				glActiveTexture(GL_TEXTURE1);
				glBindTexture(GL_TEXTURE_1D, v.tfID);
				s.setInt("tf", 1);
			}

			ImGuiFileDialog::Instance()->Close();
		}

		ImGui::End();
	}
}

void render() 
{
	glClearColor(0, 0, 0, 1.0f);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	model = glm::rotate(angleX, glm::vec3(1.0f, 0.0f, 0.0f));
	model *= glm::rotate(angleY, glm::vec3(0.0f, 1.0f, 0.0f));
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

	glEnable(GL_DEPTH_TEST);

	projection = glm::perspective(1.57f, (GLfloat)windowWidth / windowHeight, 0.1f, 100.f);
	view = glm::lookAt(eyePos,
				       glm::vec3(0.0f, 0.0f, 0.0f),
					   glm::vec3(0.0f, 1.0f, 0.0f));
}

void reshape(int w, int h)
{
	windowWidth = w, windowHeight = h;

	glViewport(0, 0, windowWidth, windowHeight);
	projection = glm::perspective(1.57f, (GLfloat)windowWidth / windowHeight, 0.1f, 100.f);

	s.setVec2("screen", windowWidth, windowHeight);

	ImGui_ImplGLUT_ReshapeFunc(w, h);
}

void keyboard(unsigned char key, int x, int y) 
{
	switch (key) 
	{
		case ' ':
			autoRotate = !autoRotate;	
			break; 
		default:
			break;
	}

	ImGui_ImplGLUT_KeyboardFunc(key, x, y);

}

bool doMove = false;
int oldX, oldY;
void mouse(int button, int state, int x, int y)
{
	switch(button)
	{
		case GLUT_LEFT_BUTTON:
			doMove = state == GLUT_DOWN;
			oldX = x, oldY = y;
			break;
		default:
			break;
	}

	ImGui_ImplGLUT_MouseFunc(button, state, x, y);
}

void mouseWheel(int button, int dir, int x, int y)
{
	zoom += dir > 0 ? 0.05f : -0.05f;

	zoom = std::max(0.0f, std::min(2.0f, zoom));
	eyePos = glm::vec3(0.0f, 0.0f, 1.0f + zoom);
	view = glm::lookAt(eyePos,
					   glm::vec3(0.0f, 0.0f, 0.0f),
			  		   glm::vec3(0.0f, 1.0f, 0.0f));
	  
	ImGui_ImplGLUT_MouseWheelFunc(button, dir, x, y);
}

void motion(int x, int y)
{
	angleY += ((float)x - oldX) / windowWidth;
	angleX += ((float)y - oldY) / windowHeight;
	oldX = x, oldY = y;

	ImGui_ImplGLUT_MotionFunc(x, y);
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

	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

	ImGui::StyleColorsDark();

	ImGui_ImplGLUT_Init();
	ImGui_ImplGLUT_InstallFuncs();
	ImGui_ImplOpenGL3_Init();

	glutReshapeFunc(reshape);
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMouseWheelFunc(mouseWheel);
	glutMotionFunc(motion);

	glutMainLoop();

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGLUT_Shutdown();
	ImGui::DestroyContext();

	return 0;
}
