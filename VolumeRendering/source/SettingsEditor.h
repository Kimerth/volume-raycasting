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

	virtual void Draw();
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

	virtual void Draw();
};

class SettingsEditor
{
public:
	void Initialize();
	void draw(bool* enabled);

	static SegmentationModelSettings segmentationSettings;

private:
	std::vector<ISettings*> settings;
};
