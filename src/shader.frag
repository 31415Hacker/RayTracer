#version 300 es
precision highp float;

in  vec2 v_uv;
out vec4 outColor;

// textures & counts
uniform sampler2D u_triTex;
uniform sampler2D u_normTex;
uniform sampler2D u_nodeTex;
uniform int       u_texSize;
uniform int       u_nodeTexSize;
uniform int       u_triCount;
uniform int       u_nodeCount;

// camera & model
uniform vec3  u_cameraPos;
uniform vec3  u_cameraTarget;
uniform vec3  u_cameraUp;
uniform float u_fov;
uniform float u_aspect;
uniform mat4  u_model;

// fetch triangle vertex position
vec3 fetchPos(int i,int v){
    int idx = i*3 + v;
    ivec2 p = ivec2(idx % u_texSize, idx / u_texSize);
    return texelFetch(u_triTex, p, 0).xyz;
}
// fetch vertex normal
vec3 fetchNor(int i,int v){
    int idx = i*3 + v;
    ivec2 p = ivec2(idx % u_texSize, idx / u_texSize);
    return normalize(texelFetch(u_normTex, p, 0).xyz);
}

// fetch BVH node (2 pixels per node)
void fetchNode(int ni,
               out vec3 mn, out vec3 mx,
               out int left, out int rc)
{
    int pix0 = ni*2;
    ivec2 p0 = ivec2(pix0 % u_nodeTexSize, pix0 / u_nodeTexSize);
    vec4 d0  = texelFetch(u_nodeTex, p0, 0);
    int pix1 = pix0 + 1;
    ivec2 p1 = ivec2(pix1 % u_nodeTexSize, pix1 / u_nodeTexSize);
    vec4 d1  = texelFetch(u_nodeTex, p1, 0);

    mn   = d0.rgb;
    left = int(d0.a + 0.5);            // child index or triStart
    mx   = d1.rgb;
    rc   = int(d1.a + (d1.a < 0.0 ? -0.5 : 0.5)); // child index or –count
}

// AABB intersection test (slab method)
bool intersectAABB(vec3 ro, vec3 invRd, vec3 mn, vec3 mx,
                   out float tmin, out float tmax)
{
    vec3 t0 = (mn - ro) * invRd;
    vec3 t1 = (mx - ro) * invRd;
    vec3 tmin3 = min(t0, t1);
    vec3 tmax3 = max(t0, t1);
    tmin = max(max(tmin3.x, tmin3.y), tmin3.z);
    tmax = min(min(tmax3.x, tmax3.y), tmax3.z);
    return tmax >= max(tmin, 0.0);
}

// Möller–Trumbore triangle‐intersection
bool triIntersect(vec3 ro, vec3 rd,
                  vec3 v0, vec3 v1, vec3 v2,
                  out float t, out float u, out float v)
{
    const float EPS = 1e-6;
    vec3 e1 = v1 - v0;
    vec3 e2 = v2 - v0;
    vec3 p  = cross(rd, e2);
    float a = dot(e1, p);
    if (abs(a) < EPS) return false;
    float invA = 1.0 / a;
    vec3 s = ro - v0;
    u = invA * dot(s, p);
    if (u < 0.0 || u > 1.0) return false;
    vec3 q = cross(s, e1);
    v = invA * dot(rd, q);
    if (v < 0.0 || u + v > 1.0) return false;
    float tt = invA * dot(e2, q);
    if (tt > EPS) {
        t = tt;
        return true;
    }
    return false;
}

bool traverseBVH(in vec3 ro, in vec3 rd, in vec3 invRd, out float bestT, out vec3 bestN) {
    const int MAX_ST = 64;
    int stack[MAX_ST];
    int sp = 0;
    stack[sp++] = 0;

    bestT = 1e20;
    bestN = vec3(0.0);

    while (sp > 0) {
        int ni = stack[--sp];
        vec3 mn, mx;
        int left, rc;
        fetchNode(ni, mn, mx, left, rc);

        float tmin, tmax;
        if (!intersectAABB(ro, invRd, mn, mx, tmin, tmax) || tmin > bestT) continue;

        if (rc < 0) {
            int start = left;
            int cnt = -rc;
            for (int j = 0; j < cnt; ++j) {
                int ti = start + j;
                vec3 v0 = fetchPos(ti, 0);
                vec3 v1 = fetchPos(ti, 1);
                vec3 v2 = fetchPos(ti, 2);
                vec3 n0 = fetchNor(ti, 0);
                vec3 n1 = fetchNor(ti, 1);
                vec3 n2 = fetchNor(ti, 2);

                float t, u, v;
                if (triIntersect(ro, rd, v0, v1, v2, t, u, v) && t < bestT) {
                    bestT = t;
                    bestN = normalize(n0 * (1.0 - u - v) + n1 * u + n2 * v);
                }
            }
        } else {
            vec3 mn0, mx0, mn1, mx1;
            int tmp;
            fetchNode(left, mn0, mx0, tmp, tmp);
            fetchNode(rc, mn1, mx1, tmp, tmp);

            float t0min, t0max, t1min, t1max;
            bool hit0 = intersectAABB(ro, invRd, mn0, mx0, t0min, t0max) && t0min <= bestT;
            bool hit1 = intersectAABB(ro, invRd, mn1, mx1, t1min, t1max) && t1min <= bestT;

            if (hit0 && hit1) {
                bool firstIsRight = t1min < t0min;
                stack[sp++] = firstIsRight ? left : rc;
                stack[sp++] = firstIsRight ? rc   : left;
            } else if (hit0) {
                stack[sp++] = left;
            } else if (hit1) {
                stack[sp++] = rc;
            }
        }
    }

    return bestT < 1e19;
}

vec3 raytrace(vec2 uv) {
    // build camera ray in world space
    vec3 forward = normalize(u_cameraTarget - u_cameraPos);
    vec3 right   = normalize(cross(forward, u_cameraUp));
    vec3 up      = cross(right, forward);

    uv = uv * 2.0 - 1.0;
    uv.x *= u_aspect;
    float fl = 1.0 / tan(u_fov * 0.5);
    vec3 ro_w = u_cameraPos;
    vec3 rd_w = normalize(forward * fl + right * uv.x + up * uv.y);

    // transform ray into object space
    mat3 invR = transpose(mat3(u_model));
    vec3 ro    = invR * ro_w;
    vec3 rd    = normalize(invR * rd_w);
    vec3 invRd = 1.0 / rd;
    vec3 light = invR * vec3(0.0, 5.0, 5.0);;

    float bestT;
    vec3 bestN;

    if (traverseBVH(ro, rd, invRd, bestT, bestN)) {
        vec3 hit = ro + bestT * rd;
        vec3 L = normalize(light - hit);
        float diff = max(dot(bestN, L), 0.0);
        return vec3(diff);
    }

    return vec3(0.6, 0.8, 1.0); // Background
}

void main() {
    outColor = vec4(raytrace(v_uv), 1.0);
}