#include "SettingsEditor.h"

SegmentationModelSettings SettingsEditor::segmentationSettings;
StyleSettings SettingsEditor::styleSettings;
InputSettings SettingsEditor::inputSettings;

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
	buf->append("\n");
}

static void* Style_ReadOpen(ImGuiContext*, ImGuiSettingsHandler*, const char* name)
{
	return (void*)&SettingsEditor::styleSettings;
}

static void Style_ReadLine(ImGuiContext*, ImGuiSettingsHandler*, void* entry, const char* line)
{
	StyleSettings* settings = (StyleSettings*)entry;

	int idx;
	float c_r, c_g, c_b, c_a;

	if(sscanf(line, "Colors[%d]=(%f, %f, %f, %f)", &idx, &c_r, &c_g, &c_b, &c_a) == 5)
		settings->refColors[idx] = ImVec4(c_r, c_g, c_b, c_a);
}

static void Style_WriteAll(ImGuiContext* ctx, ImGuiSettingsHandler* handler, ImGuiTextBuffer* buf)
{
	StyleSettings settings = SettingsEditor::styleSettings;

	buf->appendf("[%s][ ]\n", handler->TypeName);
	for (int i = 0; i < ImGuiCol_COUNT; ++i)
		buf->appendf("Colors[%d]=(%f, %f, %f, %f)\n", i, settings.refColors[i].x, settings.refColors[i].y, settings.refColors[i].z, settings.refColors[i].w);
	buf->append("\n");
}

static void Style_ApplyAll(ImGuiContext* ctx, ImGuiSettingsHandler*)
{
	SettingsEditor::styleSettings.styleFromRef();
}

static void* Input_ReadOpen(ImGuiContext*, ImGuiSettingsHandler*, const char* name)
{
	return (void*)&SettingsEditor::inputSettings;
}

static void Input_ReadLine(ImGuiContext*, ImGuiSettingsHandler*, void* entry, const char* line)
{
	InputSettings* settings = (InputSettings*)entry;

	float val;

	if (sscanf(line, "RotationSpeed=%f", &val) == 1)
		settings->rotationSpeed = val;
	else if (sscanf(line, "TranslationSpeed=%f", &val) == 1)
		settings->translationSpeed = val;
}

static void Input_WriteAll(ImGuiContext* ctx, ImGuiSettingsHandler* handler, ImGuiTextBuffer* buf)
{
	InputSettings settings = SettingsEditor::inputSettings;

	buf->appendf("[%s][ ]\n", handler->TypeName);
	buf->appendf("RotationSpeed=%f\n", settings.rotationSpeed);
	buf->appendf("TranslationSpeed=%f\n", settings.translationSpeed);
	buf->append("\n");
}

void SettingsEditor::Initialize()
{
	ImGuiContext& g = *GImGui;

	ImGuiSettingsHandler segmentation_ini_handler;
	segmentation_ini_handler.TypeName = "Segmentation";
	segmentation_ini_handler.TypeHash = ImHashStr("Segmentation");
	segmentation_ini_handler.ReadOpenFn = Segmentation_ReadOpen;
	segmentation_ini_handler.ReadLineFn = Segmentation_ReadLine;
	segmentation_ini_handler.WriteAllFn = Segmentation_WriteAll;
	g.SettingsHandlers.push_back(segmentation_ini_handler);

	ImGuiSettingsHandler file_dialog_ini_handler;
	file_dialog_ini_handler.TypeName = "FileDialog";
	file_dialog_ini_handler.TypeHash = ImHashStr("FileDialog");
	file_dialog_ini_handler.ReadOpenFn = [](ImGuiContext*, ImGuiSettingsHandler*, const char* name) { return (void*)new std::string(); };
	file_dialog_ini_handler.ReadLineFn = FileDialog_ReadLine;
	file_dialog_ini_handler.WriteAllFn = FileDialog_WriteAll;
	g.SettingsHandlers.push_back(file_dialog_ini_handler);

	ImGuiSettingsHandler style_ini_handler;
	style_ini_handler.TypeName = "Style";
	style_ini_handler.TypeHash = ImHashStr("Style");
	style_ini_handler.ReadOpenFn = Style_ReadOpen;
	style_ini_handler.ReadLineFn = Style_ReadLine;
	style_ini_handler.ApplyAllFn = Style_ApplyAll;
	style_ini_handler.WriteAllFn = Style_WriteAll;
	g.SettingsHandlers.push_back(style_ini_handler);

	ImGuiSettingsHandler input_ini_handler;
	input_ini_handler.TypeName = "Input";
	input_ini_handler.TypeHash = ImHashStr("Input");
	input_ini_handler.ReadOpenFn = Input_ReadOpen;
	input_ini_handler.ReadLineFn = Input_ReadLine;
	input_ini_handler.WriteAllFn = Input_WriteAll;
	g.SettingsHandlers.push_back(input_ini_handler);

	styleSettings.refFromStyle();

	settings.push_back(&inputSettings);
	settings.push_back(&segmentationSettings);
	settings.push_back(&styleSettings);
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
	
	settings[selected]->draw();

	ImGui::EndChild();

	ImGui::End();
}

void ISettings::draw()
{
	ImGui::Text(name);
	ImGui::SameLine();
	if(ImGui::Button("Save All"))
		ImGui::SaveIniSettingsToDisk(ImGui::GetIO().IniFilename);
	
	ImGui::Separator();
}

void SegmentationModelSettings::draw()
{
	ISettings::draw();

	ImGui::InputInt3("Input Size", inputSize);
	ImGui::InputInt3("Patch Size", patchSize);
}

void StyleSettings::draw()
{
	ISettings::draw();

	ImGuiStyle& style = ImGui::GetStyle();

	if (ImGui::Combo("Style Selector", &styleIdx, "Dark\0Light\0Classic\0"))
	{
		switch (styleIdx)
		{
		case 0:
			ImGui::StyleColorsDark();
			break;
		case 1:
			ImGui::StyleColorsLight();
			break;
		case 2:
			ImGui::StyleColorsClassic();
			break;
		}

		refFromStyle();
	}

	ImGui::Separator();

	for (int i = 0; i < ImGuiCol_COUNT; ++i)
	{
		const char* name = ImGui::GetStyleColorName(i);

		ImGui::PushID(name);

		ImGui::ColorEdit4(name, (float*)&style.Colors[i], ImGuiColorEditFlags_AlphaBar | ImGuiColorEditFlags_AlphaPreviewHalf);

		if (memcmp(&style.Colors[i], &refColors[i], sizeof(ImVec4)) != 0)
		{
			ImGui::SameLine(0.0f, style.ItemInnerSpacing.x);
			if (ImGui::Button("Save")) 
				refColors[i] = style.Colors[i];
			ImGui::SameLine(0.0f, style.ItemInnerSpacing.x);
			if (ImGui::Button("Revert"))
				style.Colors[i] = refColors[i];
		}

		ImGui::PopID();
	}

	ImGui::Separator();

	if (ImGui::Button("Save")) 
		refFromStyle();
	ImGui::SameLine();
	if (ImGui::Button("Revert"))
		styleFromRef();
}

void InputSettings::draw()
{
	ISettings::draw();

	float rot_degrees = rotationSpeed * (180.0 / IM_PI);
	if (ImGui::SliderFloat("Rotation Speed (deg/sec)", &rot_degrees, 10.0f, 90.0f, "%.1f"))
		rotationSpeed = rot_degrees * (IM_PI / 180.0);

	ImGui::SliderFloat("Translation Speed", &translationSpeed, 0.01, 0.5, "%.2f");
}
