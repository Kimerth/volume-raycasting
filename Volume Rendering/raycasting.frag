#version 400

#define MAX_PASSES 1000
#define STEP 0.01f

in vec3 start;
in vec4 exit;

uniform sampler2D tex;
uniform sampler3D volumeTex;  
uniform float power;
uniform vec2 screen;
out vec4 fragColor;

void main()
{
    vec3 exitPoint = texture(tex, gl_FragCoord.st / screen).xyz;
    
    vec3 dir = exitPoint - start;
    float maxLen = length(dir);
    vec3 deltaDir = normalize(dir) * STEP;
    float deltaDirLen = length(deltaDir);
    vec3 voxelCoord = start;
    fragColor = vec4(0);
    float len = 0.0;
    float intensity;
 
    for(int i = 0; i < MAX_PASSES; ++i, len += deltaDirLen, voxelCoord += deltaDir)
    {
    	intensity = texture(volumeTex, voxelCoord).x;
    	if (intensity > 0.0) 
        {
    	    intensity = 1.0 - pow(1.0 - intensity, power);
    	    fragColor += (1.0 - fragColor.a) * intensity;
    	}
    	
    	if (len >= maxLen )
    	    break;  
    }
}