#include "SettingsEditor.h"


static void* Segmentation_ReadOpen(ImGuiContext*, ImGuiSettingsHandler*, const char* name)
{
	return (void*)&SettingsEditor::segmentationSettings;
}

static void Segmentation_ReadLine(ImGuiContext*, ImGuiSettingsHandler*, void* entry, const char* line)
{
	SettingsEditor::SegmentationModelSettings *settings = (SettingsEditor::SegmentationModelSettings*)entry;
	
	int size[3];
	if (sscanf(line, "InputSize=%i,%i,%i", &size[0], &size[1], &size[2]) == 3)
		memcpy(settings->inputSize, size, 3 * sizeof(int));
	else if (sscanf(line, "PatchSize=%i,%i,%i", &size[0], &size[1], &size[2]) == 3)\
		memcpy(settings->patchSize, size, 3 * sizeof(int));
}

static void Segmentation_WriteAll(ImGuiContext* ctx, ImGuiSettingsHandler* handler, ImGuiTextBuffer* buf)
{
	SettingsEditor::SegmentationModelSettings settings = SettingsEditor::segmentationSettings;
	
	// buf->reserve(buf->size() + 4); // ballpark reserve
	
	buf->appendf("[%s]\n", handler->TypeName);
	buf->appendf("InputSize=%d,%d,%d\n", settings.inputSize[0], settings.inputSize[1], settings.inputSize[2]);
	buf->appendf("PatchSize=%d,%d,%d\n", settings.patchSize[0], settings.patchSize[1], settings.patchSize[2]);
	buf->append("\n");
}


void SettingsEditor::Initialize()
{
	ImGuiContext& g = *GImGui;

	ImGuiSettingsHandler segmentation_ini_handler;
	segmentation_ini_handler.TypeName = "Segmentation";
	segmentation_ini_handler.TypeHash = ImHash("Segmentation", 0);
	segmentation_ini_handler.ReadOpenFn = Segmentation_ReadOpen;
	segmentation_ini_handler.ReadLineFn = Segmentation_ReadLine;
	segmentation_ini_handler.WriteAllFn = Segmentation_WriteAll;
	g.SettingsHandlers.push_back(segmentation_ini_handler);
}
