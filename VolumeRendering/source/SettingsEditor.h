#pragma once

#include <vector>
#include <string>
#include <imgui/imgui.h>
#include <imgui/imgui_internal.h>
#include <imguiFD/ImGuiFileDialog.h>


class ISettings
{
public:
	const char* name;
	
	ISettings(const char* name) : name(name) {}

	virtual void draw();
};

class SegmentationModelSettings: public ISettings
{
public:
	int inputSize[3];
	int patchSize[3];

	SegmentationModelSettings() : ISettings("Segmentation Model")
	{
		memset(inputSize, 0, 3 * sizeof(int));
		memset(patchSize, 0, 3 * sizeof(int));
	}

	virtual void draw();
};

class StyleSettings : public ISettings
{
public:
	ImVec4 refColors[ImGuiCol_COUNT];
	int styleIdx;

	StyleSettings() : ISettings("Style")
	{
		styleIdx = -1;
	}

	void refFromStyle()
	{
		ImGuiStyle& style = ImGui::GetStyle();
		for (int i = 0; i < ImGuiCol_COUNT; ++i)
			refColors[i] = ImVec4(style.Colors[i]);
	}

	void styleFromRef()
	{
		ImGuiStyle& style = ImGui::GetStyle();
		for (int i = 0; i < ImGuiCol_COUNT; ++i)
			style.Colors[i] = ImVec4(refColors[i]);
	}

	virtual void draw();
};

class InputSettings : public ISettings
{
public:
	float rotationSpeed;
	float translationSpeed;

	InputSettings() : ISettings("Input")
	{
		rotationSpeed = 1.0;
		translationSpeed = 0.1;
	}

	virtual void draw();
};

class SettingsEditor
{
public:
	void Initialize();
	void draw(bool* enabled);

	static SegmentationModelSettings segmentationSettings;
	static StyleSettings styleSettings;
	static InputSettings inputSettings;
private:
	std::vector<ISettings*> settings;
};
