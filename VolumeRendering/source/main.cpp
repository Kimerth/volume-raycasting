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

#include <imguiFD/ImGuiFileDialog.h>

#include "Volume.h"
#include "Shader.h"
#include "Loader.h"
#include "transfer_function_widget.h"
#include "PytorchModel.h"

#define FRAG_SHADER_PATH "source/shaders/raycasting.frag"
#define VERT_SHADER_PATH "source/shaders/raycasting.vert"
#define COMP_SHADER_PATH "source/shaders/grads.comp"

#define PYTORCH_SEGMANTATION_MODULE_PATH "segmentation_model.pt"

#define FRAME_DURATON 32
#define ANGLE_SPEED 0.68 // approx 40degrees/sec

int windowWidth = 1240, windowHeight = 800;

Volume v;
Shader s;
PytorchModel ptModel;
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
	s.load(VERT_SHADER_PATH, FRAG_SHADER_PATH, COMP_SHADER_PATH, v);
	s.use();

	s.setVec2("screen", windowWidth, windowHeight);
	s.setVec3("scale", v.scale);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_3D, v.texID);
	s.setInt("volumeTex", 0);

	glActiveTexture(GL_TEXTURE2);
	glBindTexture(GL_TEXTURE_3D, v.segID);
	s.setInt("segTex", 2);

	glActiveTexture(GL_TEXTURE3);
	glBindTexture(GL_TEXTURE_3D, v.gradsID);
	s.setInt("gradsTex", 3);

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
			show_volume_window = !show_volume_window;

		if (ImGui::MenuItem("Show TF Window"))
			show_tf_window = !show_tf_window;
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
				ImGuiFileDialog::Instance()->OpenDialog("ChooseVolumeOpen", "Choose Volume", ".nii,.nii.gz", ".");
			if (ImGui::MenuItem("Load segmentation"))
				ImGuiFileDialog::Instance()->OpenDialog("ChooseSegmentationOpen", "Choose Segmentation", ".nii,.nii.gz", ".");

			ImGui::EndMenuBar();
		}

		if (ImGuiFileDialog::Instance()->Display("ChooseVolumeOpen"))
		{
			if (ImGuiFileDialog::Instance()->IsOk())
			{
				std::string filePathName = ImGuiFileDialog::Instance()->GetFilePathName();
				v.load(filePathName.c_str());

				std::string savPath = (std::string(filePathName) + ".sav");
				if (std::fstream{ savPath })
				{
					float* buffer = readTF(savPath.c_str());
					tfWidget.loadTF(buffer);
				}
			}

			ImGuiFileDialog::Instance()->Close();
		}

		if (ImGuiFileDialog::Instance()->Display("ChooseSegmentationOpen"))
		{
			if (ImGuiFileDialog::Instance()->IsOk())
			{
				std::string filePathName = ImGuiFileDialog::Instance()->GetFilePathName();
				v.loadSegmentation(filePathName.c_str());
			}

			ImGuiFileDialog::Instance()->Close();
		}

		ImGui::Text("Size: %dx%dx%d 8 bit", v.sizeX, v.sizeY, v.sizeZ);
		ImGui::PlotHistogram("Histogram", v.hist, (1 << 16), 0, NULL, 0.0f, 1.0f, ImVec2(0, 100.0f), sizeof(float));

		ImGui::End();
	}

	if (show_tf_window)
	{
		ImGui::Begin("Transfer Function", &show_volume_window, ImGuiWindowFlags_MenuBar);
		if (ImGui::BeginMenuBar())
		{
			if (ImGui::MenuItem("New"))
				tfWidget.reset();

			if (ImGui::BeginMenu("File"))
			{
				if (ImGui::MenuItem("Open.."))
					ImGuiFileDialog::Instance()->OpenDialog("ChoseTFOpen", "Choose TF", ".sav,.txt", ".");
				if (ImGui::MenuItem("Save"))
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
				float* tf = readTF(filePathName.c_str());

				tfWidget.loadTF(tf);

				glActiveTexture(GL_TEXTURE1);
				glBindTexture(GL_TEXTURE_1D, v.tfID);
				s.setInt("tf", 1);
			}

			ImGuiFileDialog::Instance()->Close();
		}

		if (ImGuiFileDialog::Instance()->Display("ChoseTFSave"))
		{
			if (ImGuiFileDialog::Instance()->IsOk())
			{
				std::string filePathName = ImGuiFileDialog::Instance()->GetFilePathName();

				std::ofstream f(filePathName);

				for (int i = 0; i < 256; ++i)
				{
					f << "re=" << tfWidget.current_colormap[4 * i] << std::endl
						<< "ge=" << tfWidget.current_colormap[4 * i + 1] << std::endl
						<< "be=" << tfWidget.current_colormap[4 * i + 2] << std::endl
						<< "ra=" << tfWidget.current_colormap[4 * i + 3] << std::endl
						<< "ga=" << tfWidget.current_colormap[4 * i + 3] << std::endl
						<< "ba=" << tfWidget.current_colormap[4 * i + 3] << std::endl;
				}
				
				f.close();
			}

			ImGuiFileDialog::Instance()->Close();
		}


		ImGui::End();
	}
}

void render() 
{
	v.loadTF(tfWidget.current_colormap);

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

	if (v.vao != NULL)
	{
		glBindVertexArray(v.vao);
		glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, (GLuint*)NULL);
	}
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

	ptModel.loadModel(PYTORCH_SEGMANTATION_MODULE_PATH);
}

void reshape(int w, int h)
{
	windowWidth = w, windowHeight = h;

	glViewport(0, 0, windowWidth, windowHeight);
	projection = glm::perspective(1.57f, (GLfloat)windowWidth / windowHeight, 0.1f, 100.f);

	s.setVec2("screen", windowWidth, windowHeight);

	ImGui_ImplGLUT_ReshapeFunc(w, h);
}

void mouseWheel(int button, int dir, int x, int y);

void keyboard(unsigned char key, int x, int y) 
{
	switch (key) 
	{
		case ' ':
			autoRotate = !autoRotate;	
			break;
		case 'w':
			mouseWheel(0, 1, 0, 0);
			break;
		case 's':
			mouseWheel(0, -1, 0, 0);
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

	zoom = std::max(-0.75f, std::min(1.0f, zoom));
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
	glutInitWindowSize(windowWidth, windowHeight);
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
