#include "transfer_function_widget.h"
#include <algorithm>
#include <cmath>
#include <iostream>

#ifndef TFN_WIDGET_NO_STB_IMAGE_IMPL
#define STB_IMAGE_IMPLEMENTATION
#endif

template <typename T>
inline T clamp(T x, T min, T max)
{
	if (x < min) 
		return min;
	if (x > max) 
		return max;
	return x;
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

TransferFunctionWidget::vec2f::vec2f(float c) : x(c), y(c) {}

TransferFunctionWidget::vec2f::vec2f(float c, int idx) : x(c), y(c), idx(idx) {}

TransferFunctionWidget::vec2f::vec2f(float x, float y) : x(x), y(y) {}

TransferFunctionWidget::vec2f::vec2f(const ImVec2& v) : x(v.x), y(v.y) {}

TransferFunctionWidget::vec2f::vec2f(const vec2f& v, int idx): x(v.x), y(v.y), idx(idx) {}

float TransferFunctionWidget::vec2f::length() const
{
	return std::sqrt(x * x + y * y);
}

TransferFunctionWidget::vec2f TransferFunctionWidget::vec2f::operator+(
	const TransferFunctionWidget::vec2f& b) const
{
	return vec2f(x + b.x, y + b.y);
}

TransferFunctionWidget::vec2f TransferFunctionWidget::vec2f::operator-(
	const TransferFunctionWidget::vec2f& b) const
{
	return vec2f(x - b.x, y - b.y);
}

TransferFunctionWidget::vec2f TransferFunctionWidget::vec2f::operator/(
	const TransferFunctionWidget::vec2f& b) const
{
	return vec2f(x / b.x, y / b.y);
}

TransferFunctionWidget::vec2f TransferFunctionWidget::vec2f::operator*(
	const TransferFunctionWidget::vec2f& b) const
{
	return vec2f(x * b.x, y * b.y);
}

TransferFunctionWidget::vec2f::operator ImVec2() const
{
	return ImVec2(x, y);
}

void TransferFunctionWidget::draw_ui()
{
	update_colormap();
	update_gpu_image();

	const ImGuiIO& io = ImGui::GetIO();

	ImGui::Text("Transfer Function");
	ImGui::TextWrapped(
		"Left click to add a point, right click remove. "
		"Left click + drag to move points.");

	if (last_point != -1)
		ImGui::ColorEdit3("Point Color", &color_points[last_point * 3]);

	vec2f canvas_size = ImGui::GetContentRegionAvail();
	size_t tmp = colormap_img;
	ImGui::Image(reinterpret_cast<void*>(tmp), ImVec2(canvas_size.x, 16));
	vec2f canvas_pos = ImGui::GetCursorScreenPos();
	canvas_size.y -= 20;

	const float point_radius = 10.f;

	ImDrawList* draw_list = ImGui::GetWindowDrawList();
	draw_list->PushClipRect(canvas_pos, canvas_pos + canvas_size);

	const vec2f view_scale(canvas_size.x, -canvas_size.y);
	const vec2f view_offset(canvas_pos.x, canvas_pos.y + canvas_size.y);

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
		vec2f mouse_pos = (vec2f(clipped_mouse_pos) - view_offset) / view_scale;
		mouse_pos.x = clamp(mouse_pos.x, 0.f, 1.f);
		mouse_pos.y = clamp(mouse_pos.y, 0.f, 1.f);

		if (io.MouseDown[0]) 
		{
			if (selected_point != (size_t)-1) 
			{
				alpha_control_pts[selected_point] = vec2f(mouse_pos, alpha_control_pts[selected_point].idx);

				// Keep the first and last control points at the edges
				if (selected_point == 0) 
					alpha_control_pts[selected_point].x = 0.f;
				else if (selected_point == alpha_control_pts.size() - 1) 
					alpha_control_pts[selected_point].x = 1.f;
			}
			else {
				auto fnd = std::find_if(
					alpha_control_pts.begin(), alpha_control_pts.end(), [&](const vec2f& p) 
					{
						const vec2f pt_pos = p * view_scale + view_offset;
						float dist = (pt_pos - vec2f(clipped_mouse_pos)).length();
						return dist <= point_radius;
					});
				// No nearby point, we're adding a new one
				if (fnd == alpha_control_pts.end()) 
				{
					alpha_control_pts.push_back(vec2f(mouse_pos, color_points.size()));
					color_points.insert(color_points.end(), {1, 1, 1});
				}
			}

			// Keep alpha control points ordered by x coordinate, update
			// selected point index to match
			std::sort(alpha_control_pts.begin(),
				alpha_control_pts.end(),
				[](const vec2f& a, const vec2f& b) { return a.x < b.x; });

			for (int i = 0; i < alpha_control_pts.size(); ++i)
			{
				vec2f v = alpha_control_pts[i];
				if (v.idx != 3 * i) 
				{
					std::swap_ranges(color_points.begin() + i * 3,
						color_points.begin() + (i + 1) * 3,
						color_points.begin() + v.idx);
					std::swap(alpha_control_pts[i].idx, alpha_control_pts[v.idx / 3].idx);
				}
			}

			if (selected_point != 0 && selected_point != alpha_control_pts.size() - 1) 
			{
				auto fnd = std::find_if(
					alpha_control_pts.begin(), alpha_control_pts.end(), [&](const vec2f& p) 
					{
						const vec2f pt_pos = p * view_scale + view_offset;
						float dist = (pt_pos - vec2f(clipped_mouse_pos)).length();
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
				alpha_control_pts.begin(), alpha_control_pts.end(), [&](const vec2f& p) 
				{
					const vec2f pt_pos = p * view_scale + view_offset;
					float dist = (pt_pos - vec2f(clipped_mouse_pos)).length();
					return dist <= point_radius;
				});
			// We also want to prevent erasing the first and last points
			if (fnd != alpha_control_pts.end() && fnd != alpha_control_pts.begin() &&
				fnd != alpha_control_pts.end() - 1)
				alpha_control_pts.erase(fnd);
		}
		else 
			selected_point = -1;
	}
	else 
		selected_point = -1;

	// Draw the alpha control points, and build the points for the polyline
	// which connects them
	std::vector<ImVec2> polyline_pts;
	for (const auto& pt : alpha_control_pts) {
		const vec2f pt_pos = pt * view_scale + view_offset;
		polyline_pts.push_back(pt_pos);
		draw_list->AddCircleFilled(pt_pos, point_radius, 0xFFFFFFFF);
	}
	draw_list->AddPolyline(
		polyline_pts.data(), (int)polyline_pts.size(), 0xFFFFFFFF, false, 2.f);
	draw_list->PopClipRect();
}

void TransferFunctionWidget::update_gpu_image()
{
	GLint prev_tex_2d = 0;
	glGetIntegerv(GL_TEXTURE_BINDING_2D, &prev_tex_2d);

	if (colormap_img == (GLuint)-1) {
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

void TransferFunctionWidget::update_colormap()
{
	int a_idx = 0;
	for (size_t i = 0; i < 256; ++i) {
		float x = static_cast<float>(i) / 256;
		if (x > alpha_control_pts[a_idx + 1].x) 
			++a_idx;
		vec2f first = alpha_control_pts[a_idx], second = alpha_control_pts[a_idx + 1];

		float t = (x - first.x) / (second.x - first.x);

		float r = (1.f - t) * color_points[3 * a_idx]	  + t * color_points[3 * (a_idx + 1)];
		float g = (1.f - t) * color_points[3 * a_idx + 1] + t * color_points[3 * (a_idx + 1) + 1];
		float b = (1.f - t) * color_points[3 * a_idx + 2] + t * color_points[3 * (a_idx + 1) + 2];
		float alpha = (1.f - t) * first.y + t * second.y;

		current_colormap[i * 4]		= clamp(r, 0.f, 1.f);
		current_colormap[i * 4 + 1] = clamp(g, 0.f, 1.f);
		current_colormap[i * 4 + 2] = clamp(b, 0.f, 1.f);
		current_colormap[i * 4 + 3] = clamp(alpha, 0.f, 1.f);
	}
}