#pragma once

#include <imgui/imgui.h>
#include <imgui/imgui_internal.h>

class SettingsEditor
{
public:
	void Initialize();

	struct SegmentationModelSettings
	{
		int inputSize[3];
		int patchSize[3];
	};

	static SegmentationModelSettings segmentationSettings;
};
