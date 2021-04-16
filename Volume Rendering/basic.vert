#version 400

in vec3 vp;
in vec3 vc;

out vec3 clr;

uniform mat4 modelMatrix;

void main()
{
    clr = vc;
    gl_Position = modelMatrix * vec4(vp, 1.0);
}