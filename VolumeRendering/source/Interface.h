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

#include "SettingsEditor.h"
#include "VectorColorPicker.h"

#include "Volume.h"
#include "Loader.h"

#define FRAME_DURATION 32

class Interface
{
public:
	void initialize();
	void render(float deltaTime);

	inline float* getTFColormap() { return tfWidget.colormap; }

	inline void canLoadVolumeFunc(const std::function<bool()>& func) { canLoadVolume = func; };
	inline void segmentationAvailableFunc(const std::function<bool()>& func) { segmentationAvailable = func; };
	inline void canLoadSegmentationFunc(const std::function<bool()>& func) { canLoadSegmentation = func; };
	inline void canComputeSegmentationFunc(const std::function<bool()>& func) { canComputeSegmentation = func; };
	inline void canSmoothSegmentationFunc(const std::function<bool()>& func) { canSmoothSegmentation = func; };

	inline void computeSegmentationFunc(const std::function<void()>& func) { computeSegmentation = func; };
	inline void smoothLabelsFunc(const std::function<void(int)>& func) { smoothLabels = func; };
	inline void loadShadersFunc(const std::function<void()>& func) { loadShaders = func; };

	inline void loadVolumeFunc(const std::function<void(const char*)>& func) { loadVolume = func; };
	inline void loadSegmentationFunc(const std::function<void(const char*)>& func) { loadSegmentation = func; };
	inline void saveSegmentationFunc(const std::function<void(const char*)>& func) { saveSegmentation = func; };
	inline void applySegmentationFunc(const std::function<void()>& func) { applySegmentation = func; };

	inline void getHistogramFunc(const std::function<float* (int nbBins)>& func) { getHistogram = func; };
	inline void getSegmentInfoFunc(const std::function<Volume::SegmentInfo* ()>& func) { getSegmentInfo = func; };

	void keyboard(unsigned char key, int x, int y);
	void specialInput(int key, int x, int y);
	void mouse(int button, int state, int x, int y);
	void mouseWheel(int button, int dir, int x, int y);
	void motion(int x, int y);

	float angleY = 3.14f, angleX = 1.57f, angleZ = 0.0f;
	float translationX, translationY, translationZ;
	float zoom = 0.5f;

	float intensityCorrection = 0.1;
	float exposure = 1, gamma = 1;
	int sampleRate = 100;

	bool autoRotate = false;

	glm::vec3 bbLow = glm::vec3(-0.5);
	glm::vec3 bbHigh = glm::vec3(0.5);

	int windowWidth = 1240, windowHeight = 800;

private:
	void display();

	void mainMenuBar();
	void volumeWindow();
	void tfWindow();

	void volumeFileMenu();
	void tfFileMenu();

	void segmentationPropertyEditor();
	void sliceSlider(const char* label, float* min, float* max, float v_min, float v_max);

	ImGuiIO* io;

	std::function<bool()> canLoadVolume;
	std::function<bool()> segmentationAvailable;
	std::function<bool()> canLoadSegmentation;
	std::function<bool()> canComputeSegmentation;
	std::function<bool()> canSmoothSegmentation;

	std::function<void()> computeSegmentation;
	std::function<void(int)> smoothLabels;
	std::function<void()> loadShaders;

	std::function<void(const char*)> loadVolume;
	std::function<void(const char*)> loadSegmentation;
	std::function<void(const char*)> saveSegmentation;
	std::function<void()> applySegmentation;

	std::function<float* (int nbBins)> getHistogram;
	std::function<Volume::SegmentInfo* ()> getSegmentInfo;

	SettingsEditor settingsEditor;
	VectorColorPicker tfWidget;

	bool show_volume_window = true;
	bool show_tf_window = true;
	bool show_settings_editor = false;

	const char* nifti_filter = "NIfTI (*.nii.gz *.nii){(.+\.nii\.gz),.nii}";

	int smoothingRadius = 0;

	int oldX, oldY;
};

