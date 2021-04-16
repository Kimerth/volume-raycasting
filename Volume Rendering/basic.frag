#version 400

in vec3 clr;
layout (location = 0) out vec4 fragClr;


void main()
{
    fragClr = vec4(clr, 1.0);
}
