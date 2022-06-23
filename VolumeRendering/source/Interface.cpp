#include "Interface.h"


void Interface::initialize()
{
	io = &ImGui::GetIO();

	io->ConfigWindowsMoveFromTitleBarOnly = true;
	io->ConfigWindowsResizeFromEdges = true;
	io->ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

	settingsEditor.Initialize();
}

void Interface::render(float deltaTime)
{
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGLUT_NewFrame();

	displayUI();
	ImGui::Render();

	if (autoRotate)
		angleY += ROTATION_SPEED * deltaTime / 1000;
}

void Interface::displayUI()
{
	if (ImGui::BeginMainMenuBar())
	{
		if (ImGui::BeginMenu("Windows"))
		{
			ImGui::Checkbox("Show Volume Window", &show_volume_window);
			ImGui::Checkbox("Show TF Window", &show_tf_window);
			ImGui::Checkbox("Show Settings Editor Window", &show_settings_editor);

			ImGui::EndMenu();
		}

		if (ImGui::MenuItem("Reload Shaders"))
			loadShaders();

		ImGui::EndMainMenuBar();
	}

	if (show_volume_window)
	{
		ImGui::Begin("Volume", &show_volume_window, ImGuiWindowFlags_MenuBar);
		if (ImGui::BeginMenuBar())
		{
			if (ImGui::BeginMenu("File"))
			{
				if (ImGui::MenuItem("Load Volume"))
					ImGuiFileDialog::Instance()->OpenDialog("ChooseVolumeLoad", "Choose Volume", nifti_filter, ".");

				if (ImGui::BeginMenu("Segmentation"))
				{
					if (ImGui::MenuItem("Load segmentation", 0, false, canLoadSegmentation()))
						ImGuiFileDialog::Instance()->OpenDialog("ChooseSegmentationLoad", "Choose Segmentation", nifti_filter, ".");

					if (ImGui::MenuItem("Save segmentation", 0, false, segmentationAvailable()))
						ImGuiFileDialog::Instance()->OpenDialog("ChooseSegmentationSave", "Choose Segmentation", nifti_filter, ".", 1, nullptr, ImGuiFileDialogFlags_ConfirmOverwrite);

					ImGui::EndMenu();
				}

				ImGui::EndMenu();
			}
			if (ImGui::MenuItem("Compute segmentation", NULL, false, canComputeSegmentation()))
			{
				computeSegmentation();
				loadShaders();
			}

			ImGui::EndMenuBar();
		}

		if (ImGuiFileDialog::Instance()->Display("ChooseVolumeLoad"))
		{
			if (ImGuiFileDialog::Instance()->IsOk())
			{
				std::string filePathName = ImGuiFileDialog::Instance()->GetFilePathName();
				loadVolume(filePathName.c_str());
				loadShaders();

				std::string savPath = (std::string(filePathName) + ".sav");
				if (std::fstream{ savPath })
					tfWidget.save(savPath.c_str());

				if (tfWidget.hist != nullptr)
					delete[] tfWidget.hist;
				tfWidget.hist = getHistogram(tfWidget.nb_bins);
			}

			ImGuiFileDialog::Instance()->Close();
		}

		if (ImGuiFileDialog::Instance()->Display("ChooseSegmentationLoad"))
		{
			if (ImGuiFileDialog::Instance()->IsOk())
			{
				std::string filePathName = ImGuiFileDialog::Instance()->GetFilePathName();
				loadSegmentation(filePathName.c_str());
				loadShaders();
			}

			ImGuiFileDialog::Instance()->Close();
		}

		if (ImGuiFileDialog::Instance()->Display("ChooseSegmentationSave"))
		{
			if (ImGuiFileDialog::Instance()->IsOk())
			{
				std::string filePathName = ImGuiFileDialog::Instance()->GetFilePathName();
				saveSegmentation(filePathName.c_str());
			}

			ImGuiFileDialog::Instance()->Close();
		}

		if (ImGui::CollapsingHeader("Slicing"))
		{
			sliceSlider("x", &bbLow.x, &bbHigh.x, -0.5, 0.5);
			sliceSlider("y", &bbLow.y, &bbHigh.y, -0.5, 0.5);
			sliceSlider("z", &bbLow.z, &bbHigh.z, -0.5, 0.5);
		}

		if (segmentationAvailable())
		{
			if (!canSmoothSegmentation())
				ImGui::BeginDisabled();

			ImGui::SliderInt("Label smoothing radius", &smoothingRadius, 0, 3);
			if (ImGui::Button("Apply label smoothing"))
				smoothLabels(smoothingRadius);

			if (!canSmoothSegmentation())
				ImGui::EndDisabled();

			if (ImGui::CollapsingHeader("Semantic segmentation"))
			{
				segmentationPropertyEditor();
			}
		}

		ImGui::End();
	}

	if (show_tf_window)
	{
		ImGui::Begin("Visualization", &show_volume_window, ImGuiWindowFlags_MenuBar);
		if (ImGui::BeginMenuBar())
		{
			if (ImGui::MenuItem("New"))
				tfWidget.reset();

			if (ImGui::BeginMenu("File"))
			{
				if (ImGui::MenuItem("Open"))
					ImGuiFileDialog::Instance()->OpenDialog("ChoseTFOpen", "Choose TF", ".sav", ".");
				if (ImGui::MenuItem("Save"))
					ImGuiFileDialog::Instance()->OpenDialog("ChoseTFSave", "Choose TF", ".sav", ".", 1, nullptr, ImGuiFileDialogFlags_ConfirmOverwrite);
				ImGui::EndMenu();
			}

			ImGui::EndMenuBar();
		}

		ImGui::SliderInt("Sample Rate", &sampleRate, 50, 300);
		ImGui::SliderFloat("Intenisty Correction", &intensityCorrection, 0.01, 1, "%.2f");
		ImGui::SliderFloat("Exposure", &exposure, 1, 10, "%.1f");
		ImGui::SliderFloat("Gamma", &gamma, 0.75, 1.25, "%.2f");

		tfWidget.draw();

		if (ImGuiFileDialog::Instance()->Display("ChoseTFOpen"))
		{
			if (ImGuiFileDialog::Instance()->IsOk())
				tfWidget.load(ImGuiFileDialog::Instance()->GetFilePathName().c_str());

			ImGuiFileDialog::Instance()->Close();
		}

		if (ImGuiFileDialog::Instance()->Display("ChoseTFSave"))
		{
			if (ImGuiFileDialog::Instance()->IsOk())
				tfWidget.save(ImGuiFileDialog::Instance()->GetFilePathName().c_str());

			ImGuiFileDialog::Instance()->Close();
		}

		ImGui::End();
	}

	if (show_settings_editor)
		settingsEditor.draw(&show_settings_editor);
}

void Interface::segmentationPropertyEditor()
{
	std::string labels[] = { "background", "liver", "bladder", "lungs", "kidneys", "bone", "brain" };

	if (ImGui::BeginTable("seg_edit", 2, ImGuiTableFlags_BordersOuter | ImGuiTableFlags_Resizable))
	{
		for (int i = 0; i < 7; i++)
		{
			ImGui::PushID(i);

			ImGui::TableNextRow();
			ImGui::TableSetColumnIndex(0);
			bool is_open = ImGui::TreeNode("Segment", labels[i].c_str());
			ImGui::TableSetColumnIndex(1);
			ImGui::Text("%d voxels", getSegmentInfo()[i].numVoxels);

			if (is_open)
			{
				std::map<std::string, std::function<void()>> rows = {
					{"Enabled", [&]() {
						if (ImGui::Checkbox(labels[i].c_str(), &getSegmentInfo()[i].enabled))
							applySegmentation();
					}},
					{"Color", [&]() {
						ImGui::ColorEdit3("", getSegmentInfo()[i].color);
					}}
				};

				for (const auto& [label, row] : rows)
				{
					ImGui::PushID(0);

					ImGui::TableNextRow();
					ImGui::TableSetColumnIndex(0);
					ImGui::TreeNodeEx("Field", ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_NoTreePushOnOpen, label.c_str());

					ImGui::TableSetColumnIndex(1);
					row();

					ImGui::NextColumn();
					ImGui::PopID();
				}

				ImGui::TreePop();
			}

			ImGui::PopID();
		}

		ImGui::EndTable();
	}
}

void Interface::sliceSlider(const char* label, float* min, float* max, float v_min, float v_max)
{
	ImGuiWindow* window = ImGui::GetCurrentWindow();
	if (window->SkipItems)
		return;

	ImGuiContext& g = *GImGui;
	const ImGuiStyle& style = g.Style;
	const ImGuiID id = window->GetID(label);
	const float w = ImGui::CalcItemWidth();

	const ImVec2 label_size = ImGui::CalcTextSize(label, NULL, true);
	const ImRect frame_bb(window->DC.CursorPos, ImVec2(window->DC.CursorPos.x + w, window->DC.CursorPos.y + label_size.y + style.FramePadding.y * 2.0f));
	const ImRect total_bb(frame_bb.Min, ImVec2(frame_bb.Max.x + style.ItemInnerSpacing.x + label_size.x, frame_bb.Max.y));

	ImGui::ItemAdd(total_bb, id);
	ImGui::ItemSize(total_bb, style.FramePadding.y);

	bool hovered = g.HoveredWindow == window && ImGui::IsMouseHoveringRect(frame_bb.Min, frame_bb.Max);
	if (hovered)
		ImGui::SetHoveredID(id);

	if (hovered && g.IO.MouseClicked[0])
	{
		ImGui::SetActiveID(id, window);
		ImGui::FocusWindow(window);
	}
	else if (g.ActiveId == id && !g.IO.MouseDown[0])
	{
		ImGui::SetActiveID(0, NULL);
		ImGui::FocusWindow(NULL);
	}

	ImGui::RenderFrame(frame_bb.Min, frame_bb.Max, ImGui::GetColorU32(ImGuiCol_FrameBg), style.FrameBorderSize > 0, style.FrameRounding);

	const float grab_padding = 2.0f;
	const float slider_size = frame_bb.GetWidth() - grab_padding * 2;
	const float grab_size = std::min(style.GrabMinSize, slider_size);
	const float slider_usable_size = slider_size - grab_size;

	if (g.ActiveId == id)
	{
		float clicked_t = ImClamp((g.IO.MousePos.x - frame_bb.Min.x) / slider_usable_size, 0.0f, 1.0f);
		float new_value = ImLerp(v_min, v_max, clicked_t);

		if (abs(*min - new_value) < abs(*max - new_value))
			*min = new_value;
		else
			*max = new_value;
	}

	float grab_t = (*min - v_min) / (v_max - v_min);
	float grab_pos = ImLerp(frame_bb.Min.x, frame_bb.Max.x, grab_t);
	ImRect grab_bb1 = ImRect(ImVec2(grab_pos - grab_size * 0.5f, frame_bb.Min.y + grab_padding), ImVec2(grab_pos + grab_size * 0.5f, frame_bb.Max.y - grab_padding));
	window->DrawList->AddRectFilled(grab_bb1.Min, grab_bb1.Max, ImGui::GetColorU32(g.ActiveId == id ? ImGuiCol_SliderGrabActive : ImGuiCol_SliderGrab), style.GrabRounding);

	grab_t = (*max - v_min) / (v_max - v_min);
	grab_pos = ImLerp(frame_bb.Min.x, frame_bb.Max.x, grab_t);
	ImRect grab_bb2 = ImRect(ImVec2(grab_pos - grab_size * 0.5f, frame_bb.Min.y + grab_padding), ImVec2(grab_pos + grab_size * 0.5f, frame_bb.Max.y - grab_padding));
	window->DrawList->AddRectFilled(grab_bb2.Min, grab_bb2.Max, ImGui::GetColorU32(g.ActiveId == id ? ImGuiCol_SliderGrabActive : ImGuiCol_SliderGrab), style.GrabRounding);

	ImRect connector(grab_bb1.Min, grab_bb2.Max);
	connector.Min.x += grab_size;
	connector.Min.y += grab_size * 0.3f;
	connector.Max.x -= grab_size;
	connector.Max.y -= grab_size * 0.3f;

	window->DrawList->AddRectFilled(connector.Min, connector.Max, ImGui::GetColorU32(ImGuiCol_SliderGrab), style.GrabRounding);

	char value_buf[64];
	const char* value_buf_end = value_buf + ImFormatString(value_buf, IM_ARRAYSIZE(value_buf), "%.2f %.2f", *min, *max);
	ImGui::RenderTextClipped(frame_bb.Min, frame_bb.Max, value_buf, value_buf_end, NULL, ImVec2(0.5f, 0.5f));

	ImGui::RenderText(ImVec2(frame_bb.Max.x + style.ItemInnerSpacing.x * 2, frame_bb.Min.y + style.FramePadding.y), label);
}

void Interface::keyboard(unsigned char key, int x, int y)
{
	if (io != nullptr && io->WantCaptureKeyboard)
	{
		ImGui_ImplGLUT_KeyboardFunc(key, x, y);
		return;
	}

	if (glutGetModifiers() == GLUT_ACTIVE_ALT)
		switch (key)
		{
		case '+':
		case 'w':
			translationZ += TRANSLATION_SPEED / FRAME_DURATION;
			break;
		case '-':
		case 's':
			translationZ -= TRANSLATION_SPEED / FRAME_DURATION;
			break;
		default:
			break;
		}
	else
		switch (key)
		{
		case ' ':
			autoRotate = !autoRotate;
			break;
		case '+':
		case 'w':
			mouseWheel(0, 1, 0, 0);
			break;
		case '-':
		case 's':
			mouseWheel(0, -1, 0, 0);
			break;
		default:
			break;
		}

	ImGui_ImplGLUT_KeyboardFunc(key, x, y);
}

void Interface::specialInput(int key, int x, int y)
{
	if (glutGetModifiers() == GLUT_ACTIVE_ALT)
		switch (key)
		{
		case GLUT_KEY_UP:
			translationY += TRANSLATION_SPEED / FRAME_DURATION;
			break;
		case GLUT_KEY_DOWN:
			translationY -= TRANSLATION_SPEED / FRAME_DURATION;
			break;
		case GLUT_KEY_LEFT:
			translationX += TRANSLATION_SPEED / FRAME_DURATION;
			break;
		case GLUT_KEY_RIGHT:
			translationX -= TRANSLATION_SPEED / FRAME_DURATION;
			break;
		default:
			break;
		}
	else
		switch (key)
		{
		case GLUT_KEY_UP:
			angleX += ROTATION_SPEED / FRAME_DURATION;
			break;
		case GLUT_KEY_DOWN:
			angleX -= ROTATION_SPEED / FRAME_DURATION;
			break;
		case GLUT_KEY_LEFT:
			angleY -= ROTATION_SPEED / FRAME_DURATION;
			break;
		case GLUT_KEY_RIGHT:
			angleY += ROTATION_SPEED / FRAME_DURATION;
			break;
		default:
			break;
		}

	ImGui_ImplGLUT_SpecialFunc(key, x, y);
}

void Interface::mouse(int button, int state, int x, int y)
{
	if (io != nullptr && io->WantCaptureMouse)
	{
		ImGui_ImplGLUT_MouseFunc(button, state, x, y);
		return;
	}

	switch (button)
	{
	case GLUT_LEFT_BUTTON:
		oldX = x, oldY = y;
		break;
	default:
		break;
	}

	ImGui_ImplGLUT_MouseFunc(button, state, x, y);
}

void Interface::mouseWheel(int button, int dir, int x, int y)
{
	if (io != nullptr && !io->WantCaptureMouse)
		zoom += dir > 0 ? 0.05f : -0.05f;

	ImGui_ImplGLUT_MouseWheelFunc(button, dir, x, y);
}

void Interface::motion(int x, int y)
{
	if (io != nullptr && !io->WantCaptureMouse)
	{
		angleY += ((float)x - oldX) / windowWidth;
		angleX += ((float)y - oldY) / windowHeight;
		oldX = x, oldY = y;
	}

	ImGui_ImplGLUT_MotionFunc(x, y);
}
