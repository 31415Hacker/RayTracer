// ------------------------------
// shader.wgsl
// ------------------------------

struct Uniforms {
    cameraPos:  vec4<f32>,    // world-space camera pos
    lightPos:   vec4<f32>,    // world-space light center (w unused)
    camRight:   vec4<f32>,    // camera basis (world-space)
    camUp:      vec4<f32>,
    camForward: vec4<f32>,
    model:      mat4x4<f32>,  // world→object transform (inverse), pre-inverted in JS
};

@group(0) @binding(0) var<storage,read> triPos0:   array<vec4<f32>>;
@group(0) @binding(1) var<storage,read> triPos1:   array<vec4<f32>>;
@group(0) @binding(2) var<storage,read> triNor0:   array<vec4<f32>>;
@group(0) @binding(3) var<storage,read> triNor1:   array<vec4<f32>>;
@group(0) @binding(4) var<storage,read> bvhNodes0: array<vec4<f32>>;
@group(0) @binding(5) var<storage,read> bvhNodes1: array<vec4<f32>>;
@group(0) @binding(6) var<uniform>    uniforms:   Uniforms;

// Intermediate types
struct RayData { ro: vec3<f32>, rd: vec3<f32>, invRd: vec3<f32> };
struct Hit     { t: f32, n: vec3<f32>        };
struct VSOut   { @builtin(position) Position: vec4<f32>, @location(0) uv: vec2<f32> };

// Vertex shader: full-screen quad
@vertex
fn vs_main(@location(0) pos: vec2<f32>) -> VSOut {
    var o: VSOut;
    o.Position = vec4<f32>(pos, 0.0, 1.0);
    o.uv = pos * 0.5 + vec2<f32>(0.5);
    return o;
}

// SSBO fetch helpers
fn fetchTriPos(idx: u32) -> vec3<f32> {
    let C = __COUNT_POS0__;
    if (idx < C) { return triPos0[idx].xyz; }
    return triPos1[idx - C].xyz;
}
fn fetchTriNor(idx: u32) -> vec3<f32> {
    let C = __COUNT_NOR0__;
    if (idx < C) { return triNor0[idx].xyz; }
    return triNor1[idx - C].xyz;
}
fn fetchBVHNode(idx: u32) -> vec4<f32> {
    let C = __COUNT_BVH0__;
    if (idx < C) { return bvhNodes0[idx]; }
    return bvhNodes1[idx - C];
}

// AABB intersection
fn intersectAABB(
    ro: vec3<f32>, invRd: vec3<f32>,
    mn: vec3<f32>, mx: vec3<f32>,
    tminOut: ptr<function,f32>, tmaxOut: ptr<function,f32>
) -> bool {
    let t0 = (mn - ro) * invRd;
    let t1 = (mx - ro) * invRd;
    let tmin3 = min(t0, t1);
    let tmax3 = max(t0, t1);
    let tmin  = max(max(tmin3.x, tmin3.y), tmin3.z);
    let tmax  = min(min(tmax3.x, tmax3.y), tmax3.z);
    *tminOut = tmin; *tmaxOut = tmax;
    return tmax >= max(tmin, 0.0);
}

// Möller–Trumbore triangle intersection
fn triIntersect(
    ro: vec3<f32>, rd: vec3<f32>,
    v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>,
    tOut: ptr<function,f32>, uOut: ptr<function,f32>, vOut: ptr<function,f32>
) -> bool {
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let p  = cross(rd, e2);
    let a  = dot(e1, p);
    if (abs(a) < 0.0) { return false; }
    let invA = 1.0 / a;
    let s    = ro - v0;
    let u    = invA * dot(s, p);
    if (u < 0.0 || u > 1.0) { return false; }
    let q = cross(s, e1);
    let v = invA * dot(rd, q);
    if (v < 0.0 || u + v > 1.0) { return false; }
    let t = invA * dot(e2, q);
    if (t > 0.0) {
        *tOut = t; *uOut = u; *vOut = v;
        return true;
    }
    return false;
}

fn isSafeVec(v: vec3<f32>) -> bool {
    return all(abs(v) <= vec3<f32>(1e30));
}

// Generate world-space ray direction
fn generateRay(uv: vec2<f32>) -> vec3<f32> {
    let scr = uv * 2.0 - vec2<f32>(1.0);
    return normalize(
        uniforms.camRight.xyz   * scr.x +
        uniforms.camUp.xyz      * scr.y +
        uniforms.camForward.xyz
    );
}

// World→object transform (use pre-inverted matrix)
fn toObjectSpace(ro_w: vec3<f32>, rd_w: vec3<f32>) -> RayData {
    let ro4 = uniforms.model * vec4<f32>(ro_w, 1.0);
    let R3  = mat3x3<f32>(
        uniforms.model[0].xyz,
        uniforms.model[1].xyz,
        uniforms.model[2].xyz
    );
    let rdObj = normalize(R3 * rd_w);
    return RayData(ro4.xyz, rdObj, 1.0 / rdObj);
}

// Stack-based BVH traversal, returns nearest Hit in object space
fn traverseBVH(ro: vec3<f32>, rd: vec3<f32>, invRd: vec3<f32>) -> Hit {
    var stack: array<i32,32>;
    var sp:    i32 = 1;
    stack[0]   = 0;
    var bestT  = 1e20;
    var bestN  = vec3<f32>(0.0);

    loop {
        if (sp == 0) { break; }
        sp -= 1;
        let ni = stack[sp];
        let i0 = u32(ni * 2);
        let b0 = fetchBVHNode(i0);
        let b1 = fetchBVHNode(i0 + 1u);
        let mn = b0.xyz;
        let mx = b1.xyz;

        var tmin: f32; var tmax: f32;
        if (!intersectAABB(ro, invRd, mn, mx, &tmin, &tmax) || tmin > bestT) {
            continue;
        }

        let rightFlag = b1.w;
        if (rightFlag < 0.0) {
            // leaf
            let start = i32(b0.w);
            let cnt   = -i32(rightFlag);
            for (var j = 0; j < cnt; j = j + 1) {
                let ti = u32(start + j);
                let v0 = fetchTriPos(ti*3u + 0u);
                let v1 = fetchTriPos(ti*3u + 1u);
                let v2 = fetchTriPos(ti*3u + 2u);
                let geoN = cross(v1 - v0, v2 - v0);
                if (dot(geoN, rd) > 0.0) { continue; }
                var t: f32; var u: f32; var v: f32;
                if (triIntersect(ro, rd, v0, v1, v2, &t, &u, &v) && t < bestT) {
                    bestT = t;
                    let n0 = fetchTriNor(ti*3u + 0u);
                    let n1 = fetchTriNor(ti*3u + 1u);
                    let n2 = fetchTriNor(ti*3u + 2u);
                    bestN = normalize(n0*(1.0-u-v) + n1*u + n2*v);
                }
            }
        } else {
            // internal: push children
            let c0 = i32(b0.w);
            let c1 = i32(rightFlag);
            var t0min: f32; var t0max: f32;
            var t1min: f32; var t1max: f32;
            let mn0 = fetchBVHNode(u32(c0*2)).xyz;
            let mx0 = fetchBVHNode(u32(c0*2)+1u).xyz;
            let mn1 = fetchBVHNode(u32(c1*2)).xyz;
            let mx1 = fetchBVHNode(u32(c1*2)+1u).xyz;
            let hit0 = intersectAABB(ro, invRd, mn0, mx0, &t0min, &t0max) && t0min <= bestT;
            let hit1 = intersectAABB(ro, invRd, mn1, mx1, &t1min, &t1max) && t1min <= bestT;
            if (hit0 && hit1) {
                if (t0min <= t1min) {
                    if (sp < 32) { stack[sp] = c1; sp += 1; }
                    if (sp < 32) { stack[sp] = c0; sp += 1; }
                } else {
                    if (sp < 32) { stack[sp] = c0; sp += 1; }
                    if (sp < 32) { stack[sp] = c1; sp += 1; }
                }
            } else if (hit0) {
                if (sp < 32) { stack[sp] = c0; sp += 1; }
            } else if (hit1) {
                if (sp < 32) { stack[sp] = c1; sp += 1; }
            }
        }
    }

    return Hit(bestT, bestN);
}

// constant for sampling
const PI = 3.141592653589793;
const SOFT_SAMPLES = 4;

// simple 2D hash → [0,1)
fn rand2(seed: vec2<f32>) -> f32 {
    return fract(sin(dot(seed, vec2<f32>(12.9898,78.233))) * 43758.5453);
}

/// Stratified soft‐shadow: returns [0,1] light visibility
fn computeSoftShadow(hit_ws: vec3<f32>, n_ws: vec3<f32>) -> f32 {
    // build an orthonormal basis (T,B) around the true light direction
    let L = normalize(uniforms.lightPos.xyz - hit_ws);
    var up = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(L.y) > 0.99) { up = vec3<f32>(1.0, 0.0, 0.0); }
    let T = normalize(cross(up, L));
    let B = cross(L, T);

    var occ: f32 = 0.0;
    for (var i: i32 = 0; i < SOFT_SAMPLES; i = i + 1) {
        let fi = f32(i);
        // stratified seeds
        let seed = hit_ws.xy * (fi + 1.37) + hit_ws.yz * (fi + 2.49);
        let a = 2.0 * PI * rand2(seed);
        let r = sqrt(rand2(seed.yx));
        // sample point on disc
        let offset = (T * cos(a) + B * sin(a)) * (r * uniforms.lightPos.w);
        let samplePos = uniforms.lightPos.xyz + offset;

        // trace toward that point
        let dir_ws    = normalize(samplePos - hit_ws);
        let origin_ws = hit_ws + n_ws * 0.001;
        let rdObj     = toObjectSpace(origin_ws, dir_ws);
        let sh        = traverseBVH(rdObj.ro, rdObj.rd, rdObj.invRd);

        // if something blocks before reaching sample
        if (sh.t < length(samplePos - hit_ws)) {
            occ = occ + 1.0;
        }
    }
    // visibility = 1 - occlusion_fraction
    return 1.0 - occ / f32(SOFT_SAMPLES);
}

/// Lambert diffuse + soft shadows
fn computeLighting(ro: vec3<f32>, rd: vec3<f32>, hit: Hit) -> vec4<f32> {
    if (hit.t < 1e19) {
        // object-space → world-space hit & normal
        let R3     = mat3x3<f32>(
            uniforms.model[0].xyz,
            uniforms.model[1].xyz,
            uniforms.model[2].xyz
        );
        let RT     = transpose(R3);               // original model
        let hit_os = ro + hit.t * rd;
        let hit_ws = RT * hit_os;                 // world-space point
        var Nws    = normalize(RT * hit.n);       // world-space normal
        if (!isSafeVec(Nws)) {
            Nws = vec3<f32>(0.0,1.0,0.0);
        }

        // Lambert term
        let L    = normalize(uniforms.lightPos.xyz - hit_ws);
        let diff = max(dot(Nws, L), 0.0);

        // soft‐shadow factor
        let vis  = computeSoftShadow(hit_ws, Nws);

        let c = diff * vis;
        return vec4<f32>(c, c, c, 1.0);
    }
    // sky
    return vec4<f32>(0.6, 0.8, 1.0, 1.0);
}

// Raytrace entry
fn raytrace(uv: vec2<f32>) -> vec4<f32> {
    let ro_w = uniforms.cameraPos.xyz;
    let rd_w = generateRay(uv);
    let rdObj= toObjectSpace(ro_w, rd_w);
    let hit  = traverseBVH(rdObj.ro, rdObj.rd, rdObj.invRd);
    return computeLighting(rdObj.ro, rdObj.rd, hit);
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    return raytrace(in.uv);
}
