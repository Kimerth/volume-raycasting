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
#include "Loader.h"

#define COLORMAP_SIZE 256

enum ColorSpace { LINEAR, SRGB };

class VectorColorPicker {

    std::vector<float> colors = { 1, 1, 1, 1, 1, 1, };
    std::vector<ImVec2> points = { ImVec2(0.f, 1.f), ImVec2(1.f, 1.f)};
    size_t selected = -1;
    size_t previous = -1;

    bool clicked = false;
    GLuint colormapTex;

public:
    float colormap[COLORMAP_SIZE * 4];
    const int nbBins = 128;
    float* hist;

    void draw();
    void reset();
    void load(const char* path);
    void save(const char* path);

private:
    void updateColormapTexture();
    void updateColormap();

    const float point_radius = 10.f;
};

