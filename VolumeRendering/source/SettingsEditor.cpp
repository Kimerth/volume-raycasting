#include "SettingsEditor.h"

SegmentationModelSettings SettingsEditor::segmentationSettings;

static void* Segmentation_ReadOpen(ImGuiContext*, ImGuiSettingsHandler*, const char* name)
{
	return (void*)&SettingsEditor::segmentationSettings;
}

static void Segmentation_ReadLine(ImGuiContext*, ImGuiSettingsHandler*, void* entry, const char* line)
{
	SegmentationModelSettings *settings = (SegmentationModelSettings*)entry;
	
	int size[3];
	if (sscanf(line, "InputSize=%i,%i,%i", &size[0], &size[1], &size[2]) == 3)
		memcpy(settings->inputSize, size, 3 * sizeof(int));
	else if (sscanf(line, "PatchSize=%i,%i,%i", &size[0], &size[1], &size[2]) == 3)
		memcpy(settings->patchSize, size, 3 * sizeof(int));
}

static void Segmentation_WriteAll(ImGuiContext* ctx, ImGuiSettingsHandler* handler, ImGuiTextBuffer* buf)
{
	SegmentationModelSettings settings = SettingsEditor::segmentationSettings;
	
	buf->appendf("[%s][ ]\n", handler->TypeName);
	buf->appendf("InputSize=%d,%d,%d\n", settings.inputSize[0], settings.inputSize[1], settings.inputSize[2]);
	buf->appendf("PatchSize=%d,%d,%d\n", settings.patchSize[0], settings.patchSize[1], settings.patchSize[2]);
	buf->append("\n");
}

static void FileDialog_ReadLine(ImGuiContext*, ImGuiSettingsHandler*, void* entry, const char* line)
{
	std::string* bookmarks = (std::string*)entry;

	char buffer[256] = { 0 };

	if (sscanf(line, "Bookmarks=%s", buffer) == 1)
	{
		bookmarks->assign(buffer);
		ImGuiFileDialog::Instance()->DeserializeBookmarks(buffer);
	}
}

static void FileDialog_WriteAll(ImGuiContext* ctx, ImGuiSettingsHandler* handler, ImGuiTextBuffer* buf)
{
	buf->appendf("[%s][ ]\n", handler->TypeName);
	buf->appendf("Bookmarks=%s\n", ImGuiFileDialog::Instance()->SerializeBookmarks().c_str());
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

	ImGuiSettingsHandler file_dialog_ini_handler;
	file_dialog_ini_handler.TypeName = "FileDialog";
	file_dialog_ini_handler.TypeHash = ImHash("FileDialog", 0);
	file_dialog_ini_handler.ReadOpenFn = [](ImGuiContext*, ImGuiSettingsHandler*, const char* name) { return (void*)new std::string(); };
	file_dialog_ini_handler.ReadLineFn = FileDialog_ReadLine;
	file_dialog_ini_handler.WriteAllFn = FileDialog_WriteAll;
	g.SettingsHandlers.push_back(file_dialog_ini_handler);

	settings.push_back(&segmentationSettings);
}

void SettingsEditor::draw(bool *enabled)
{
	ImGui::Begin("Settings", enabled);
	
	static int selected = 0;
	ImGui::BeginChild("left", ImVec2(150, 0), true);
	for (int i = 0; i < settings.size(); i++)
	{
		if (ImGui::Selectable(settings[i]->name, selected == i))
			selected = i;
	}
	ImGui::EndChild();
	ImGui::SameLine();
	
	ImGui::BeginChild("right");
	
	settings[selected]->Draw();

	ImGui::EndChild();

	ImGui::End();
}

void ISettings::Draw()
{
	ImGui::Text(name);
	ImGui::Separator();
}

void SegmentationModelSettings::Draw()
{
	ISettings::Draw();

	ImGui::InputInt3("Input Size", inputSize);
	ImGui::InputInt3("Patch Size", patchSize);
}
