#include "Interface.h"


void Interface::render(float deltaTime)
{
	ImGui_ImplOpenGL3_NewFrame();
	ImGui_ImplGLUT_NewFrame();

	displayUI();
	ImGui::Render();
	io = &ImGui::GetIO();

	io->ConfigWindowsMoveFromTitleBarOnly = true;
	io->ConfigWindowsResizeFromEdges = true;

	if (autoRotate)
		angleY += ROTATION_SPEED * deltaTime / 1000;
}

void Interface::displayUI()
{
	ImGui::Begin("Main", NULL, ImGuiWindowFlags_MenuBar);

	if (ImGui::BeginMenuBar())
	{
		if (ImGui::MenuItem("Show Volume Window"))
			show_volume_window = !show_volume_window;

		if (ImGui::MenuItem("Show TF Window"))
			show_tf_window = !show_tf_window;
		ImGui::EndMenuBar();
	}

	if (ImGui::Button("Reload Shaders"))
	{
		loadShaders();
	}

	if (ImGui::CollapsingHeader("Info"))
	{
		ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
	}

	ImGui::End();

	if (show_volume_window)
	{
		ImGui::Begin("Volume", &show_volume_window, ImGuiWindowFlags_MenuBar);
		if (ImGui::BeginMenuBar())
		{
			if (ImGui::MenuItem("Open"))
				ImGuiFileDialog::Instance()->OpenDialog("ChooseVolumeOpen", "Choose Volume", ".gz,.nii", ".");
			if (ImGui::MenuItem("Load segmentation"))
				ImGuiFileDialog::Instance()->OpenDialog("ChooseSegmentationOpen", "Choose Segmentation", ".gz,.nii", ".");
			if (ImGui::MenuItem("Compute segmentation", NULL, false, canComputeSegmentation()))
				computeSegmentation();

			ImGui::EndMenuBar();
		}

		if (ImGuiFileDialog::Instance()->Display("ChooseVolumeOpen"))
		{
			if (ImGuiFileDialog::Instance()->IsOk())
			{
				std::string filePathName = ImGuiFileDialog::Instance()->GetFilePathName();
				loadVolume(filePathName.c_str());
				loadShaders();

				std::string savPath = (std::string(filePathName) + ".sav");
				if (std::fstream{ savPath })
				{
					float* buffer = readTF(savPath.c_str());
					tfWidget.loadTF(buffer);
				}
			}

			ImGuiFileDialog::Instance()->Close();
		}

		if (ImGuiFileDialog::Instance()->Display("ChooseSegmentationOpen"))
		{
			if (ImGuiFileDialog::Instance()->IsOk())
			{
				std::string filePathName = ImGuiFileDialog::Instance()->GetFilePathName();
				loadSegmentation(filePathName.c_str());
			}

			ImGuiFileDialog::Instance()->Close();
		}

		if (ImGui::CollapsingHeader("Info"))
		{
			// ImGui::Text("Size: %dx%dx%d 16 bit", v.sizeX, v.sizeY, v.sizeZ);
			// ImGui::PlotHistogram("Histogram", v.hist, USHRT_MAX, 0, NULL, 0.0f, 1.0f, ImVec2(0, 100.0f), sizeof(float));
		}

		if (ImGui::CollapsingHeader("Slicing"))
		{
			ImGui::SliderFloat("-x", &bbLow.x, -0.5, 0.5);
			ImGui::SliderFloat("+x", &bbHigh.x, -0.5, 0.5);
			ImGui::SliderFloat("-y", &bbLow.y, -0.5, 0.5);
			ImGui::SliderFloat("+y", &bbHigh.y, -0.5, 0.5);
			ImGui::SliderFloat("-z", &bbLow.z, -0.5, 0.5);
			ImGui::SliderFloat("+z", &bbHigh.z, -0.5, 0.5);
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
				std::string labels[] = { "background", "liver", "bladder", "lungs", "kidneys", "bone", "brain" };

				for (int i = 0; i < 7; i++)
					if (ImGui::Checkbox(labels[i].c_str(), &getLabelsEnabled()[i]))
						applySegmentation();
			}
		}

		ImGui::End();
	}

	if (show_tf_window)
	{
		ImGui::Begin("Transfer Function", &show_volume_window, ImGuiWindowFlags_MenuBar);
		if (ImGui::BeginMenuBar())
		{
			if (ImGui::MenuItem("New"))
				tfWidget.reset();

			if (ImGui::BeginMenu("File"))
			{
				if (ImGui::MenuItem("Open"))
					ImGuiFileDialog::Instance()->OpenDialog("ChoseTFOpen", "Choose TF", ".sav,.txt", ".");
				if (ImGui::MenuItem("Save"))
					ImGuiFileDialog::Instance()->OpenDialog("ChoseTFSave", "Choose TF", ".sav,.txt", ".");
				ImGui::EndMenu();
			}

			ImGui::EndMenuBar();
		}

		ImGui::SliderInt("Sample Rate", &sampleRate, 50, 300);
		ImGui::SliderFloat("Exposure", &exposure, 1, 10, "%.1f");
		ImGui::SliderFloat("Gamma", &gamma, 0.75, 1.25, "%.2f");

		tfWidget.draw_ui();

		if (ImGuiFileDialog::Instance()->Display("ChoseTFOpen"))
		{
			if (ImGuiFileDialog::Instance()->IsOk())
			{
				std::string filePathName = ImGuiFileDialog::Instance()->GetFilePathName();
				float* tf = readTF(filePathName.c_str());
				tfWidget.loadTF(tf);

				loadTF();
			}

			ImGuiFileDialog::Instance()->Close();
		}

		if (ImGuiFileDialog::Instance()->Display("ChoseTFSave"))
		{
			if (ImGuiFileDialog::Instance()->IsOk())
			{
				std::string filePathName = ImGuiFileDialog::Instance()->GetFilePathName();

				std::ofstream f(filePathName);

				for (int i = 0; i < 256; ++i)
				{
					f << "re=" << tfWidget.current_colormap[4 * i] << std::endl
						<< "ge=" << tfWidget.current_colormap[4 * i + 1] << std::endl
						<< "be=" << tfWidget.current_colormap[4 * i + 2] << std::endl
						<< "ra=" << tfWidget.current_colormap[4 * i + 3] << std::endl
						<< "ga=" << tfWidget.current_colormap[4 * i + 3] << std::endl
						<< "ba=" << tfWidget.current_colormap[4 * i + 3] << std::endl;
				}

				f.close();
			}

			ImGuiFileDialog::Instance()->Close();
		}


		ImGui::End();
	}
}

float* Interface::getTFColormap()
{
	return tfWidget.current_colormap;
}

void Interface::keyboard(unsigned char key, int x, int y)
{
	if (io->WantCaptureKeyboard)
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
	if (io->WantCaptureMouse)
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
	if (!io->WantCaptureMouse)
		zoom += dir > 0 ? 0.05f : -0.05f;

	ImGui_ImplGLUT_MouseWheelFunc(button, dir, x, y);
}

void Interface::motion(int x, int y)
{
	if (!io->WantCaptureMouse)
	{
		angleY += ((float)x - oldX) / windowWidth;
		angleX += ((float)y - oldY) / windowHeight;
		oldX = x, oldY = y;
	}

	ImGui_ImplGLUT_MotionFunc(x, y);
}
