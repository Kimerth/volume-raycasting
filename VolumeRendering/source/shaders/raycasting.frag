#version 400

#define MAX_PASSES 1000
#define REFERENCE_SAMPLE_RATE 100

uniform sampler1D tf;
uniform sampler3D volumeTex;  
uniform sampler3D gradsTex;
uniform sampler3D segTex;
uniform sampler1D segColors;

uniform mat4 modelMatrix;
uniform mat4 viewMatrix;

uniform vec2 screen;
uniform vec3 scale;
uniform mat4 xtoi;
out vec4 fragColor;

uniform vec3 origin;

uniform int sampleRate;
uniform float intensityCorrection;
uniform float exposure;
uniform float gamma;

uniform vec3 translation;

uniform vec3 bbLow;
uniform vec3 bbHigh;

struct Ray {
    vec3 origin;
    vec3 dir;
};

struct AABB {
    vec3 min;
    vec3 max;
};

struct Intersection{
    vec3 p1;
    vec3 p2;
};

vec3 transform(vec3 v, mat4 M)
{
    return (vec4(v, 0) * M).xyz;
}

Intersection rayBBIntersection(Ray r, AABB aabb)
{
	vec3 invR = 1.0 / r.dir;
    vec3 tbot = invR * (aabb.min-r.origin);
    vec3 ttop = invR * (aabb.max-r.origin);
    vec3 tmin = min(ttop, tbot);
    vec3 tmax = max(ttop, tbot);
    vec2 t = max(tmin.xx, tmin.yz);
    float t0 = max(t.x, t.y);
    t = min(tmax.xx, tmax.yz);
    float t1 = min(t.x, t.y);

    return Intersection(transform(r.origin + r.dir * t0, xtoi) / scale + 0.5,
                        transform(r.origin + r.dir * t1, xtoi) / scale + 0.5);
}

void main()
{
    vec3 direction;
    direction.xy = 2.0 * gl_FragCoord.xy / screen - 1.0;
    direction.x *= screen.x / screen.y;
    direction.z = - 1 / tan(3.14 / 4);
    direction = transform(direction, viewMatrix);

    Ray ray = Ray(origin + translation, direction);
    AABB bb = AABB(bbLow, bbHigh);
    Intersection intersection = rayBBIntersection(ray, bb);

	float step = 1.0 / float(sampleRate);
    float opacityCorrectionFactor = float(REFERENCE_SAMPLE_RATE) / float(sampleRate);
    float rayLength = length(intersection.p2 - intersection.p1);
    vec3 deltaDir = step * normalize(intersection.p2 - intersection.p1);

    float intensityScale = rayLength * step;

    float len = 0.0;
	
    vec3 boundaryLow = bbLow + vec3(0.5);
	vec3 boundaryHigh = bbHigh + vec3(0.5);

    vec3 pos = intersection.p1;
    fragColor = vec4(0);

    for(int i = 0; 
        i < MAX_PASSES && len < rayLength && fragColor.a < 1;
        ++i, len += step, pos += deltaDir)
    {
	    if(all(greaterThan(pos, boundaryLow)) && all(lessThan(pos, boundaryHigh)))
        {
    	    float intensity = texture(volumeTex, pos).x;
            float seg = texture(segTex, pos).x;
            vec4 tfSample = texture(tf, intensity);

			if (seg > 0)
			{
			    vec3 segColor = texture(segColors, seg).rgb;
			    tfSample.rgb *= segColor;
			}
			else
                tfSample.a = 0;
            
//            vec3 grad = texture(gradsTex, pos).xyz;
//            vec3 N = normalize(viewMatrix * vec4(grad, 0)).xyz;
//            float coef = max(0.0, dot(N, N));

            intensity = max(0.0, 1 - pow((1 - intensity), opacityCorrectionFactor)) * intensityCorrection;
            fragColor += (1.0 - fragColor.a) * vec4(tfSample.rgb, 1) * intensity * tfSample.a;
        }
    }

    fragColor = min(vec4(1), fragColor);
    fragColor.rgb = vec3(1.0) - exp(-fragColor.rgb * exposure);
    fragColor = vec4(pow(fragColor.rgb, vec3(1.0 / gamma)), 1.0);
}