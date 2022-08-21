#include "VectorColorPicker.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

#ifndef TFN_WIDGET_NO_STB_IMAGE_IMPL
#define STB_IMAGE_IMPLEMENTATION
#endif


inline float length(ImVec2 v)
{
	return std::sqrt(v.x * v.x + v.y * v.y);
}

void VectorColorPicker::draw()
{
	const ImGuiIO& io = ImGui::GetIO();

	ImGui::Text("Transfer Function");

	if (previous != -1)
		ImGui::ColorEdit3("Point Color", &colors[previous * 3]);

	ImVec2 canvas_size = ImGui::GetContentRegionAvail();
	size_t tmp = colormapTex;
	ImGui::Image(reinterpret_cast<void*>(tmp), ImVec2(canvas_size.x, 16));
	ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
	canvas_size.y -= 20;

	ImDrawList* draw_list = ImGui::GetWindowDrawList();
	draw_list->PushClipRect(canvas_pos, canvas_pos + canvas_size);

	const ImVec2 view_scale(canvas_size.x, -canvas_size.y);
	const ImVec2 view_offset(canvas_pos.x, canvas_pos.y + canvas_size.y);

	draw_list->AddRect(canvas_pos, canvas_pos + canvas_size, ImGui::GetColorU32(ImGuiCol_FrameBg));

	ImGui::InvisibleButton("tfn_canvas", canvas_size);

	if (!io.MouseDown[0] && !io.MouseDown[1])
		clicked = false;
	if (ImGui::IsItemHovered() && (io.MouseDown[0] || io.MouseDown[1]))
		clicked = true;

	ImVec2 bbmin = ImGui::GetItemRectMin();
	ImVec2 bbmax = ImGui::GetItemRectMax();
	ImVec2 clipped_mouse_pos = ImVec2(std::min(std::max(io.MousePos.x, bbmin.x), bbmax.x),
		std::min(std::max(io.MousePos.y, bbmin.y), bbmax.y));

	if (clicked) {
		ImVec2 mouse_pos = (ImVec2(clipped_mouse_pos) - view_offset) / view_scale;
		mouse_pos.x = ImClamp(mouse_pos.x, 0.f, 1.f);
		mouse_pos.y = ImClamp(mouse_pos.y, 0.f, 1.f);

		if (io.MouseDown[0])
		{
			if (selected != (size_t)-1)
			{
				points[selected] = ImVec2(mouse_pos);

				if (selected == 0)
					points[selected].x = 0.f;
				else if (selected == points.size() - 1)
					points[selected].x = 1.f;
			}
			else {
				auto fnd = std::find_if(
					points.begin(), points.end(), [&](const ImVec2& p)
					{
						const ImVec2 pt_pos = p * view_scale + view_offset;
						float dist = length(pt_pos - ImVec2(clipped_mouse_pos));
						return dist <= point_radius;
					});
				
				if (fnd == points.end())
				{
					points.push_back(ImVec2(mouse_pos));
					colors.insert(colors.end(), { 1, 1, 1 });
				}
			}

			std::vector<std::size_t> perm(points.size());
			std::iota(perm.begin(), perm.end(), 0);
			std::sort(
				perm.begin(),
				perm.end(),
				[&](const std::size_t i, const std::size_t j) 
				{
					return points[i].x < points[j].x;
				}
			);

			std::vector<ImVec2> old_alpha_control_pts(points.size());
			std::copy(points.begin(), points.end(), old_alpha_control_pts.begin());

			std::transform(
				perm.begin(),
				perm.end(),
				points.begin(),
				[&](const std::size_t i) { return old_alpha_control_pts[i]; }
			);

			for (int i = 0; i < perm.size(); ++i)
				while (perm[i] != i)
				{
					std::swap_ranges(colors.begin() + i * 3,
						colors.begin() + (i + 1) * 3,
						colors.begin() + perm[i] * 3);
					for (int j = i + 1; j < perm.size(); ++j)
						if (perm[j] == i)
						{
							std::swap(perm[i], perm[j]);
							break;
						}
				}

			if (selected != 0 && selected != points.size() - 1)
			{
				auto fnd = std::find_if(
					points.begin(), points.end(), [&](const ImVec2& p)
					{
						const ImVec2 pt_pos = p * view_scale + view_offset;
						float dist = length(pt_pos - ImVec2(clipped_mouse_pos));
						return dist <= point_radius;
					});
				selected = std::distance(points.begin(), fnd);
				previous = selected;
			}
		}
		else if (ImGui::IsMouseClicked(1))
		{
			selected = -1;
			auto fnd = std::find_if(
				points.begin(), points.end(), [&](const ImVec2& p)
				{
					const ImVec2 pt_pos = p * view_scale + view_offset;
					float dist = length(pt_pos - ImVec2(clipped_mouse_pos));
					return dist <= point_radius;
				});
			size_t idx = fnd - points.begin();
			
			if (fnd != points.end() && fnd != points.begin() &&
				fnd != points.end() - 1)
			{
				colors.erase(colors.begin() + idx, colors.begin() + idx + 3);
				points.erase(fnd);
				previous = -1;
			}
		}
		else
			selected = -1;
	}
	else
		selected = -1;

	
	if (hist != nullptr)
	{
		float bin_w = canvas_size.x / nbBins;

		for (int i = 0; i < nbBins; ++i)
		{
			const ImVec2 bin_pos = ImVec2(i * bin_w, 0);
			const ImVec2 bin_size = ImVec2(bin_w, -hist[i] * canvas_size.y);

			ImVec2 pt_min = bin_pos + view_offset;
			ImVec2 pt_max = bin_pos + bin_size + view_offset;
			draw_list->AddRectFilled(pt_min, pt_max, ImGui::GetColorU32(ImGuiCol_PlotHistogram));
		}
	}

	std::vector<ImVec2> polyline_pts;
	for (const auto& pt : points) {
		const ImVec2 pt_pos = pt * view_scale + view_offset;
		polyline_pts.push_back(pt_pos);
		draw_list->AddCircleFilled(pt_pos, point_radius, 0xFFFFFFFF);
	}
	draw_list->AddPolyline(
		polyline_pts.data(), (int)polyline_pts.size(), 0xFFFFFFFF, false, 2.f);

	draw_list->PopClipRect();

	updateColormap();
	updateColormapTexture();
}

void VectorColorPicker::load(const char* path)
{
	readTF(path, points, colors);
}

void VectorColorPicker::save(const char* path)
{
	saveTF(path, points, colors);
}

void VectorColorPicker::reset()
{
	previous = -1;
	selected = -1;
	colors = { 1, 1, 1, 1, 1, 1, };
	points = { ImVec2(0.f, 1.f), ImVec2(1.f, 1.f) };
}

void VectorColorPicker::updateColormapTexture()
{
	GLint prev_tex_2d = 0;
	glGetIntegerv(GL_TEXTURE_BINDING_2D, &prev_tex_2d);

	if (colormapTex == NULL) {
		glGenTextures(1, &colormapTex);
		glBindTexture(GL_TEXTURE_2D, colormapTex);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	}

	glBindTexture(GL_TEXTURE_2D, colormapTex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, COLORMAP_SIZE, 1, 0, GL_RGBA, GL_FLOAT, colormap);

	glBindTexture(GL_TEXTURE_2D, prev_tex_2d);
}

void VectorColorPicker::updateColormap()
{
	int p_idx = 0;
	for (int i = 0; i < COLORMAP_SIZE; ++i) {
		float x = (float)i / COLORMAP_SIZE;
		if (x > points[p_idx + 1].x)
			++p_idx;

		ImVec2 first = points[p_idx], second = points[p_idx + 1];

		float t = (x - first.x) / (second.x - first.x);

		float r = (1.f - t) * colors[3 * p_idx]	  + t * colors[3 * (p_idx + 1)];
		float g = (1.f - t) * colors[3 * p_idx + 1] + t * colors[3 * (p_idx + 1) + 1];
		float b = (1.f - t) * colors[3 * p_idx + 2] + t * colors[3 * (p_idx + 1) + 2];
		float alpha = (1.f - t) * first.y + t * second.y;

		colormap[i * 4]		= ImClamp(r, 0.f, 1.f);
		colormap[i * 4 + 1] = ImClamp(g, 0.f, 1.f);
		colormap[i * 4 + 2] = ImClamp(b, 0.f, 1.f);
		colormap[i * 4 + 3] = ImClamp(alpha, 0.f, 1.f);
	}
}