#pragma once

#include <imgui/imgui.h>
#include <imgui/backends/imgui_impl_glut.h>
#include <imgui/backends/imgui_impl_opengl3.h>

#include <imguiFD/ImGuiFileDialog.h>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <glm/mat4x4.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "transfer_function_widget.h"

#include "Loader.h"


#define ROTATION_SPEED 0.68 // approx 40degrees/sec
#define TRANSLATION_SPEED 0.1
#define FRAME_DURATION 32

class Interface
{
public:
	void render(float deltaTime);

	float* getTFColormap();

	inline void segmentationAvailableFunc(bool (*func)()) { segmentationAvailable = func; };
	inline void canComputeSegmentationFunc(bool (*func)()) { canComputeSegmentation = func; };
	inline void canSmoothSegmentationFunc(bool (*func)()) { canSmoothSegmentation = func; };

	inline void computeSegmentationFunc(void (*func)()) { computeSegmentation = func; };
	inline void smoothLabelsFunc(void (*func)(int)) { smoothLabels = func; };
	inline void loadShadersFunc(void (*func)()) { loadShaders = func; };

	inline void loadVolumeFunc(void (*func)(const char*)) { loadVolume = func; };
	inline void loadTFFunc(void (*func)()) { loadTF = func; };
	inline void loadSegmentationFunc(void (*func)(const char*)) { loadSegmentation = func; };
	inline void applySegmentationFunc(void (*func)()) { applySegmentation = func; };

	inline void getlabelsEnabledFunc(bool* (*func)()) { getLabelsEnabled = func; };

	void keyboard(unsigned char key, int x, int y);
	void specialInput(int key, int x, int y);
	void mouse(int button, int state, int x, int y);
	void mouseWheel(int button, int dir, int x, int y);
	void motion(int x, int y);

	float angleY, angleX = 3.14f;
	float translationX, translationY, translationZ;
	float zoom = 0.5f;

	float exposure = 10, gamma = 1;
	int sampleRate = 100;

	bool autoRotate = false;

	glm::vec3 bbLow = glm::vec3(-0.5);
	glm::vec3 bbHigh = glm::vec3(0.5);

	int windowWidth = 1240, windowHeight = 800;

private:
	void displayUI();

	ImGuiIO* io;

	bool (*segmentationAvailable)();
	bool (*canComputeSegmentation)();
	bool (*canSmoothSegmentation)();

	void (*computeSegmentation)();
	void (*smoothLabels)(int);
	void (*loadShaders)();

	void (*loadVolume)(const char*);
	void (*loadTF)();
	void (*loadSegmentation)(const char*);
	void (*applySegmentation)();

	bool* (*getLabelsEnabled)();

	TransferFunctionWidget tfWidget;

	bool show_volume_window = true;
	bool show_tf_window = true;

	int smoothingRadius = 0;
	
	int oldX, oldY;
};

