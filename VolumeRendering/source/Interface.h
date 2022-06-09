#pragma once

#include <imgui/imgui.h>
#include <imgui/imgui_internal.h>
#include <imgui/backends/imgui_impl_glut.h>
#include <imgui/backends/imgui_impl_opengl3.h>

#include <imguiFD/ImGuiFileDialog.h>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <glm/mat4x4.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "VectorColorPicker.h"

#include "Loader.h"


#define ROTATION_SPEED 0.68 // approx 40degrees/sec
#define TRANSLATION_SPEED 0.1
#define FRAME_DURATION 32

class Interface
{
public:
	void render(float deltaTime);

	inline float* getTFColormap() { return tfWidget.current_colormap; }

	inline void segmentationAvailableFunc(const std::function<bool()>& func) { segmentationAvailable = func; };
	inline void canComputeSegmentationFunc(const std::function<bool()>& func) { canComputeSegmentation = func; };
	inline void canSmoothSegmentationFunc(const std::function<bool()>& func) { canSmoothSegmentation = func; };

	inline void computeSegmentationFunc(const std::function<void()>& func) { computeSegmentation = func; };
	inline void smoothLabelsFunc(const std::function<void(int)>& func) { smoothLabels = func; };
	inline void loadShadersFunc(const std::function<void()>& func) { loadShaders = func; };

	inline void loadVolumeFunc(const std::function<void(const char*)>& func) { loadVolume = func; };
	inline void loadTFFunc(const std::function<void()>& func) { loadTF = func; };
	inline void loadSegmentationFunc(const std::function<void(const char*)>& func) { loadSegmentation = func; };
	inline void applySegmentationFunc(const std::function<void()>& func) { applySegmentation = func; };

	inline void getHistogramFunc(const std::function<float* (int nb_bins)>& func) { getHistogram = func; };
	inline void getlabelsEnabledFunc(const std::function<bool* ()>& func) { getLabelsEnabled = func; };

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

	void sliceSlider(const char* label, float* min, float* max, float v_min, float v_max);

	ImGuiIO* io;

	std::function<bool()> segmentationAvailable;
	std::function<bool()> canComputeSegmentation;
	std::function<bool()> canSmoothSegmentation;

	std::function<void()> computeSegmentation;
	std::function<void(int)> smoothLabels;
	std::function<void()> loadShaders;

	std::function<void(const char*)> loadVolume;
	std::function<void()> loadTF;
	std::function<void(const char*)> loadSegmentation;
	std::function<void()> applySegmentation;

	std::function<float* (int nb_bins)> getHistogram;
	std::function<bool* ()> getLabelsEnabled;

	VectorColorPicker tfWidget;

	bool show_volume_window = true;
	bool show_tf_window = true;

	int smoothingRadius = 0;

	int oldX, oldY;
};

