#version 400

#define MAX_PASSES 1000
#define STEP 0.01
#define THRESHOLD 0.07

uniform sampler1D tf;
uniform sampler3D volumeTex;  
uniform mat4 viewMatrix;

uniform vec2 screen;
out vec4 fragColor;

uniform vec3 origin;

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

    return Intersection(r.origin + r.dir * t0 + 0.5,
                        r.origin + r.dir * t1 + 0.5);
}

void main()
{
    vec3 direction;
    direction.xy = 2.0 * gl_FragCoord.xy / screen - 1.0;
    direction.z = - 1 / tan(3.14 / 4);
    direction = (vec4(direction, 0) * viewMatrix).xyz;

    Ray ray = Ray(origin, direction);
    AABB bb = AABB(vec3(-0.5), vec3(0.5));
    Intersection intersection = rayBBIntersection(ray, bb);

    float rayLength = length(intersection.p2- intersection.p1);
    vec3 deltaDir = STEP * normalize(intersection.p2 - intersection.p1);

    float len = 0.0;

    vec3 pos = intersection.p1;
    fragColor = vec4(0);

    for(int i = 0; i < MAX_PASSES && len < rayLength; ++i, len += STEP, pos += deltaDir)
    {
    	float intensity = texture(volumeTex, pos).x;

        if(intensity >= THRESHOLD)
        {
            vec4 tfColor = texture(tf, intensity);
            tfColor.a = (exp(3 * tfColor.a) - 1.0) / (exp(3.0) - 1.0);
            fragColor.rgb += (1.0 - fragColor.a) * tfColor.rgb * tfColor.a;
            fragColor.a += (1.0 - fragColor.a) * tfColor.a;
        }
    }
}