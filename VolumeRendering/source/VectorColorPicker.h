#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <imgui/imgui.h>
#include <glm/glm.hpp>
#include <GL/glew.h>
#include <GL/freeglut.h>

#define IMGUI_DEFINE_MATH_OPERATORS
#include "imgui_internal.h"

enum ColorSpace { LINEAR, SRGB };

class VectorColorPicker {

    std::vector<float> color_points = { 1, 1, 1, 1, 1, 1, };

    std::vector<ImVec2> alpha_control_pts = { ImVec2(0.f, 1.f), ImVec2(1.f, 1.f)};
    size_t selected_point = -1;
    size_t last_point = -1;

    bool clicked_on_item = false;
    GLuint colormap_img;

public:
    float current_colormap[256 * 4];
    int nb_bins = 32;
    float* hist;

    void draw();
    void loadTF(float data[]);
    void reset();

private:
    void update_gpu_image();
    void update_colormap();
};

