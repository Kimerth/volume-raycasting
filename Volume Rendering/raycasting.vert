#version 400

in vec3 vp;
in vec3 vc;  

out vec3 start;
out vec4 exit;

uniform mat4 modelMatrix;

void main()
{
    start = vc;
    gl_Position = modelMatrix * vec4(vp,1.0);
    exit = gl_Position;  
}