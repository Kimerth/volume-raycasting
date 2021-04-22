#version 400

#define MAX_PASSES 1000
#define STEP 0.01
#define DEADZONE 0.07

in vec3 start;
in vec4 exit;

uniform sampler1D tf;
uniform sampler2D tex;
uniform sampler3D volumeTex;  
uniform vec2 screen;
out vec4 fragColor;

float alphaBlend(float alpha)
{
    return (exp(2 * alpha) - 1.0) / (exp(2.0) - 1.0);
}

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
 
    for(int i = 0; i < MAX_PASSES; ++i, len += deltaDirLen, voxelCoord += deltaDir)
    {
    	float intensity = texture(volumeTex, voxelCoord).x;

        if(intensity >= DEADZONE)
        {
            vec4 tfColor = texture(tf, intensity);
            tfColor.a = alphaBlend(tfColor.a);
            fragColor.rgb += (1.0 - fragColor.a) * tfColor.rgb * tfColor.a;
            fragColor.a += (1.0 - fragColor.a) * tfColor.a;
        }
    	
    	if (len >= maxLen || fragColor.a >= 1.0)
    	    break;  
    }
}