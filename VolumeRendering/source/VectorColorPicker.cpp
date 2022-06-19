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

inline float srgb_to_linear(const float x)
{
	if (x <= 0.04045f) {
		return x / 12.92f;
	}
	else {
		return std::pow((x + 0.055f) / 1.055f, 2.4f);
	}
}

void VectorColorPicker::draw()
{
	const ImGuiIO& io = ImGui::GetIO();

	ImGui::Text("Transfer Function");
	ImGui::TextWrapped(
		"Left click to add a point, right click remove. "
		"Left click + drag to move points.");

	if (last_point != -1)
		ImGui::ColorEdit3("Point Color", &color_points[last_point * 3]);

	ImVec2 canvas_size = ImGui::GetContentRegionAvail();
	size_t tmp = colormap_img;
	ImGui::Image(reinterpret_cast<void*>(tmp), ImVec2(canvas_size.x, 16));
	ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
	canvas_size.y -= 20;

	const float point_radius = 10.f;

	ImDrawList* draw_list = ImGui::GetWindowDrawList();
	draw_list->PushClipRect(canvas_pos, canvas_pos + canvas_size);

	const ImVec2 view_scale(canvas_size.x, -canvas_size.y);
	const ImVec2 view_offset(canvas_pos.x, canvas_pos.y + canvas_size.y);

	draw_list->AddRect(canvas_pos, canvas_pos + canvas_size, ImColor(180, 180, 180, 255));

	ImGui::InvisibleButton("tfn_canvas", canvas_size);

	static bool clicked_on_item = false;
	if (!io.MouseDown[0] && !io.MouseDown[1])
		clicked_on_item = false;
	if (ImGui::IsItemHovered() && (io.MouseDown[0] || io.MouseDown[1]))
		clicked_on_item = true;

	ImVec2 bbmin = ImGui::GetItemRectMin();
	ImVec2 bbmax = ImGui::GetItemRectMax();
	ImVec2 clipped_mouse_pos = ImVec2(std::min(std::max(io.MousePos.x, bbmin.x), bbmax.x),
		std::min(std::max(io.MousePos.y, bbmin.y), bbmax.y));

	if (clicked_on_item) {
		ImVec2 mouse_pos = (ImVec2(clipped_mouse_pos) - view_offset) / view_scale;
		mouse_pos.x = ImClamp(mouse_pos.x, 0.f, 1.f);
		mouse_pos.y = ImClamp(mouse_pos.y, 0.f, 1.f);

		if (io.MouseDown[0])
		{
			if (selected_point != (size_t)-1)
			{
				alpha_control_pts[selected_point] = ImVec2(mouse_pos);

				// keep the first and last control points at the edges
				if (selected_point == 0)
					alpha_control_pts[selected_point].x = 0.f;
				else if (selected_point == alpha_control_pts.size() - 1)
					alpha_control_pts[selected_point].x = 1.f;
			}
			else {
				auto fnd = std::find_if(
					alpha_control_pts.begin(), alpha_control_pts.end(), [&](const ImVec2& p)
					{
						const ImVec2 pt_pos = p * view_scale + view_offset;
						float dist = length(pt_pos - ImVec2(clipped_mouse_pos));
						return dist <= point_radius;
					});
				
				// no nearby point => add new one
				if (fnd == alpha_control_pts.end())
				{
					alpha_control_pts.push_back(ImVec2(mouse_pos));
					color_points.insert(color_points.end(), { 1, 1, 1 });
				}
			}

			std::vector<std::size_t> perm(alpha_control_pts.size());
			std::iota(perm.begin(), perm.end(), 0);
			std::sort(
				perm.begin(),
				perm.end(),
				[&](const std::size_t i, const std::size_t j) 
				{
					return alpha_control_pts[i].x < alpha_control_pts[j].x;
				}
			);

			std::vector<ImVec2> old_alpha_control_pts(alpha_control_pts.size());
			std::copy(alpha_control_pts.begin(), alpha_control_pts.end(), old_alpha_control_pts.begin());

			std::transform(
				perm.begin(),
				perm.end(),
				alpha_control_pts.begin(),
				[&](const std::size_t i) { return old_alpha_control_pts[i]; }
			);

			for (int i = 0; i < perm.size(); ++i)
				while (perm[i] != i)
				{
					std::swap_ranges(color_points.begin() + i * 3,
						color_points.begin() + (i + 1) * 3,
						color_points.begin() + perm[i] * 3);
					for (int j = i + 1; j < perm.size(); ++j)
						if (perm[j] == i)
						{
							std::swap(perm[i], perm[j]);
							break;
						}
				}

			if (selected_point != 0 && selected_point != alpha_control_pts.size() - 1)
			{
				auto fnd = std::find_if(
					alpha_control_pts.begin(), alpha_control_pts.end(), [&](const ImVec2& p)
					{
						const ImVec2 pt_pos = p * view_scale + view_offset;
						float dist = length(pt_pos - ImVec2(clipped_mouse_pos));
						return dist <= point_radius;
					});
				selected_point = std::distance(alpha_control_pts.begin(), fnd);
				last_point = selected_point;
			}
		}
		else if (ImGui::IsMouseClicked(1))
		{
			selected_point = -1;
			// Find and remove the point
			auto fnd = std::find_if(
				alpha_control_pts.begin(), alpha_control_pts.end(), [&](const ImVec2& p)
				{
					const ImVec2 pt_pos = p * view_scale + view_offset;
					float dist = length(pt_pos - ImVec2(clipped_mouse_pos));
					return dist <= point_radius;
				});
			size_t idx = fnd - alpha_control_pts.begin();
			// We also want to prevent erasing the first and last points
			if (fnd != alpha_control_pts.end() && fnd != alpha_control_pts.begin() &&
				fnd != alpha_control_pts.end() - 1)
			{
				color_points.erase(color_points.begin() + idx, color_points.begin() + idx + 3);
				alpha_control_pts.erase(fnd);
				last_point = -1;
			}
		}
		else
			selected_point = -1;
	}
	else
		selected_point = -1;

	
	if (hist != nullptr)
	{
		float bin_w = canvas_size.x / nb_bins;

		for (int i = 0; i < nb_bins; ++i)
		{
			const ImVec2 bin_pos = ImVec2(i * bin_w, 0);
			const ImVec2 bin_size = ImVec2(bin_w, -hist[i] * canvas_size.y);

			ImVec2 pt_min = bin_pos + view_offset;
			ImVec2 pt_max = bin_pos + bin_size + view_offset;
			draw_list->AddRectFilled(pt_min, pt_max, ImGui::GetColorU32(ImGuiCol_PlotHistogram));
		}
	}

	std::vector<ImVec2> polyline_pts;
	for (const auto& pt : alpha_control_pts) {
		const ImVec2 pt_pos = pt * view_scale + view_offset;
		polyline_pts.push_back(pt_pos);
		draw_list->AddCircleFilled(pt_pos, point_radius, 0xFFFFFFFF);
	}
	draw_list->AddPolyline(
		polyline_pts.data(), (int)polyline_pts.size(), 0xFFFFFFFF, false, 2.f);

	draw_list->PopClipRect();

	update_colormap();
	update_gpu_image();
}

void VectorColorPicker::loadTF(float data[])
{
	// TODO proper load and safe for TF

	//loadedFromFile = true;

	//alpha_control_pts.clear();
	//alpha_control_pts.push_back(ImVec2(0, data[3]));

	//color_points.clear();
	//color_points.insert(color_points.begin(), data, data + 3);

	//int nb = 1;
	//for (int i = 2; i < 255; ++i)
	//{
	//	alpha_control_pts.push_back(ImVec2(static_cast<float>(i) / 256, data[4 * i + 3]));
	//	color_points.insert(color_points.end(), data + 4 * i, data + 4 * i + 3);
	//}

	//alpha_control_pts.push_back(ImVec2(1, data[256 * 4 - 1]));
	//color_points.insert(color_points.begin(), data + 256 * 4 - 3, data + 256 * 4);

	//std::copy(data, data + 4 * 256, current_colormap);

	//update_gpu_image();
}

void VectorColorPicker::reset()
{
	last_point = -1;
	selected_point = -1;
	color_points = { 1, 1, 1, 1, 1, 1, };
	alpha_control_pts = { ImVec2(0.f, 0), ImVec2(1.f, 3) };
}

void VectorColorPicker::update_gpu_image()
{
	GLint prev_tex_2d = 0;
	glGetIntegerv(GL_TEXTURE_BINDING_2D, &prev_tex_2d);

	if (colormap_img == NULL) {
		glGenTextures(1, &colormap_img);
		glBindTexture(GL_TEXTURE_2D, colormap_img);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	}

	glBindTexture(GL_TEXTURE_2D, colormap_img);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, 256, 1, 0, GL_RGBA, GL_FLOAT, current_colormap);

	glBindTexture(GL_TEXTURE_2D, prev_tex_2d);
}

void VectorColorPicker::update_colormap()
{
	int a_idx = 0;
	for (size_t i = 0; i < 256; ++i) {
		float x = static_cast<float>(i) / 256;
		if (x > alpha_control_pts[a_idx + 1].x)
			++a_idx;
		ImVec2 first = alpha_control_pts[a_idx], second = alpha_control_pts[a_idx + 1];

		float t = (x - first.x) / (second.x - first.x);

		float r = (1.f - t) * color_points[3 * a_idx]	  + t * color_points[3 * (a_idx + 1)];
		float g = (1.f - t) * color_points[3 * a_idx + 1] + t * color_points[3 * (a_idx + 1) + 1];
		float b = (1.f - t) * color_points[3 * a_idx + 2] + t * color_points[3 * (a_idx + 1) + 2];
		float alpha = (1.f - t) * first.y + t * second.y;

		current_colormap[i * 4]		= ImClamp(r, 0.f, 1.f);
		current_colormap[i * 4 + 1] = ImClamp(g, 0.f, 1.f);
		current_colormap[i * 4 + 2] = ImClamp(b, 0.f, 1.f);
		current_colormap[i * 4 + 3] = ImClamp(alpha, 0.f, 1.f);
	}
}