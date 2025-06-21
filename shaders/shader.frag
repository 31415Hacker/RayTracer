#version 430 core
layout(location=0) out vec4 FragColor;
in vec2 vUV;
//uniform vec2 uResolution;
//uniform vec3 uCamPos;
//uniform mat3 uCamBasis;

struct Triangle { vec3 v0, v1, v2; };
//layout(std430, binding=0) buffer Triangles { Triangle tris[]; };
struct BVHNode { vec3 bboxMin, bboxMax; int left, right, triOffset, triCount; };
//layout(std430, binding=1) buffer BVH { BVHNode nodes[]; };

// AABB and triangle‐ray tests (same as before)
bool intersectAABB(vec3 ro, vec3 rd, vec3 mn, vec3 mx, out float tmin, out float tmax) {
    vec3 invR = 1.0/rd;
    vec3 t0s  = (mn - ro)*invR;
    vec3 t1s  = (mx - ro)*invR;
    vec3 tsm  = min(t0s, t1s);
    vec3 tb   = max(t0s, t1s);
    tmin = max(max(tsm.x, tsm.y), tsm.z);
    tmax = min(min(tb.x,  tb.y),  tb.z);
    return tmax >= max(tmin, 0.0);
}

bool intersectTri(vec3 ro, vec3 rd, Triangle tri, out float t) {
    const float EPS = 1e-8;
    vec3 e1 = tri.v1 - tri.v0;
    vec3 e2 = tri.v2 - tri.v0;
    vec3 p  = cross(rd, e2);
    float det = dot(e1, p);
    if (abs(det) < EPS) return false;
    float invDet = 1.0/det;
    vec3 tv = ro - tri.v0;
    float u = dot(tv, p)*invDet;
    if (u<0.0||u>1.0) return false;
    vec3 q = cross(tv, e1);
    float v = dot(rd, q)*invDet;
    if (v<0.0||u+v>1.0) return false;
    t = dot(e2, q)*invDet;
    return t>EPS;
}

void main() {
    FragColor = vec4(gl_FragCoord.xy, 1.0, 1.0);
}