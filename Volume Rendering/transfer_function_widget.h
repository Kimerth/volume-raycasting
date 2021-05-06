#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <imgui/imgui.h>
#include <GL/glew.h>
#include <GL/freeglut.h>

enum ColorSpace { LINEAR, SRGB };

class TransferFunctionWidget {
    struct vec2f {
        float x, y;
        int idx;

        vec2f(float c = 0.f);
        vec2f(float c, int idx);
        vec2f(float x, float y);
        vec2f(const ImVec2 &v);
        vec2f(const vec2f& v, int idx);

        float length() const;

        vec2f operator+(const vec2f &b) const;
        vec2f operator-(const vec2f &b) const;
        vec2f operator/(const vec2f &b) const;
        vec2f operator*(const vec2f &b) const;
        operator ImVec2() const;
    };

    float current_colormap[256 * 4];
    std::vector<float> color_points = { 1, 1, 1, 1, 1, 1, };

    std::vector<vec2f> alpha_control_pts = {vec2f(0.f, 0), vec2f(1.f, 3)};
    size_t selected_point = -1;
    size_t last_point = -1;

    bool clicked_on_item = false;
    GLuint colormap_img = -1;

public:
    void draw_ui();

private:
    void update_gpu_image();
    void update_colormap();
};

