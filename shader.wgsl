// — Uniforms shared by both vertex and fragment stages —
struct Uniforms {
    cameraPos:    vec4<f32>,
    cameraTarget: vec4<f32>,
    cameraUp:     vec4<f32>,
    lightPos:     vec4<f32>,
    fovAspect:    vec2<f32>,
    pad:          vec2<f32>,
    model:        mat4x4<f32>,
};

// — Storage buffers for triangle data & BVH nodes —
// triPos0/1: vec4<f32> arrays holding xyz position in .xyz
// triNor0/1: vec4<f32> arrays holding xyz normal in .xyz
// bvhNodes0/1: vec4<f32> arrays holding [mn.x,mn.y,mn.z,left] and [mx.x,mx.y,mx.z,right]
@group(0) @binding(0) var<storage, read> triPos0:   array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> triPos1:   array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> triNor0:   array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> triNor1:   array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> bvhNodes0: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> bvhNodes1: array<vec4<f32>>;

// — Uniform block —
@group(0) @binding(6) var<uniform> uniforms: Uniforms;

// — Vertex→fragment payload —
struct VSOut {
    @builtin(position) Position: vec4<f32>,
    @location(0)       uv:       vec2<f32>,
};

// — Fullscreen‐quad vertex shader →
@vertex
fn vs_main(@location(0) pos: vec2<f32>) -> VSOut {
    var out: VSOut;
    out.Position = vec4<f32>(pos, 0.0, 1.0);
    // Map from [-1,1] to [0,1]
    out.uv = pos * vec2<f32>(0.5, 0.5) + vec2<f32>(0.5, 0.5);
    return out;
}

// — Helpers to fetch from split SSBOs —
fn fetchtriPos(idx: u32) -> vec3<f32> {
    let COUNT: u32 = __COUNT_POS0__;
    if (idx < COUNT) {
        return triPos0[idx].xyz;
    } else {
        return triPos1[idx - COUNT].xyz;
    }
}
fn fetchtriNor(idx: u32) -> vec3<f32> {
    let COUNT: u32 = __COUNT_NOR0__;
    if (idx < COUNT) {
        return triNor0[idx].xyz;
    } else {
        return triNor1[idx - COUNT].xyz;
    }
}
fn fetchbvhNodes(idx: u32) -> vec4<f32> {
    let COUNT: u32 = __COUNT_BVH0__;
    if (idx < COUNT) {
        return bvhNodes0[idx];
    } else {
        return bvhNodes1[idx - COUNT];
    }
}

// — AABB intersection helper —
fn intersectAABB(
    ro:      vec3<f32>,
    invRd:   vec3<f32>,
    mn:      vec3<f32>,
    mx:      vec3<f32>,
    tminOut: ptr<function, f32>,
    tmaxOut: ptr<function, f32>,
) -> bool {
    let t0 = (mn - ro) * invRd;
    let t1 = (mx - ro) * invRd;
    let tmin3 = min(t0, t1);
    let tmax3 = max(t0, t1);
    let tmin = max(max(tmin3.x, tmin3.y), tmin3.z);
    let tmax = min(min(tmax3.x, tmax3.y), tmax3.z);
    *tminOut = tmin;
    *tmaxOut = tmax;
    return tmax >= max(tmin, 0.0);
}

// — Möller–Trumbore triangle intersection helper —
fn triIntersect(
    ro:    vec3<f32>,
    rd:    vec3<f32>,
    v0:    vec3<f32>,
    v1:    vec3<f32>,
    v2:    vec3<f32>,
    tOut:  ptr<function, f32>,
    uOut:  ptr<function, f32>,
    vOut:  ptr<function, f32>,
) -> bool {
    let epsilon = 0.0;
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let p  = cross(rd, e2);
    let a  = dot(e1, p);
    if (abs(a) < epsilon) { return false; }
    let invA = 1.0 / a;
    let s = ro - v0;
    let u = invA * dot(s, p);
    if (u < 0.0 || u > 1.0) { return false; }
    let q = cross(s, e1);
    let v = invA * dot(rd, q);
    if (v < 0.0 || u + v > 1.0) { return false; }
    let t = invA * dot(e2, q);
    if (t > epsilon) {
        *tOut = t;
        *uOut = u;
        *vOut = v;
        return true;
    }
    return false;
}

// — Safe‐check finite vector (no NaNs or infinities) —
fn isSafeVec3(v: vec3<f32>) -> bool {
    return all(abs(v) <= vec3<f32>(1e30));
}

// — Fragment: ray‐trace the BVH and shade Phong diffuse —
@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    // 1) Generate primary ray in world space
    let ro_w    = uniforms.cameraPos.xyz;
    let forward = normalize(uniforms.cameraTarget.xyz - ro_w);
    let right   = normalize(cross(forward, uniforms.cameraUp.xyz));
    let upv     = cross(right, forward);

    var uv = in.uv * vec2<f32>(2.0, 2.0) - vec2<f32>(1.0, 1.0);
    uv.x = uv.x * uniforms.fovAspect.y;
    let fl   = 1.0 / tan(uniforms.fovAspect.x * 0.5);
    let rd_w = normalize(forward * fl + right * uv.x + upv * uv.y);

    // 2) Transform ray to object (model) space
    let M3 = mat3x3<f32>(
        uniforms.model[0].xyz,
        uniforms.model[1].xyz,
        uniforms.model[2].xyz
    );
    let ro    = M3 * ro_w;
    let rd    = normalize(M3 * rd_w);
    let invRd = 1.0 / rd;

    // 3) BVH traversal using a small stack
    var stack: array<i32, 32>;
    var sp:    i32 = 1;
    stack[0] = 0;

    var bestT: f32      = 1e20;
    var bestN: vec3<f32> = vec3<f32>(0.0);

    loop {
        if (sp == 0) { break; }
        sp -= 1;
        let ni = stack[sp];

        // Fetch this node’s AABB & child‐info in two vec4 loads
        let i0    = u32(ni * 2);
        let node0 = fetchbvhNodes(i0);      // .xyz = mn, .w = left index
        let node1 = fetchbvhNodes(i0 + 1u); // .xyz = mx, .w = right or -count
        let mn    = node0.xyz;
        let mx    = node1.xyz;

        var tmin: f32; var tmax: f32;
        if (!intersectAABB(ro, invRd, mn, mx, &tmin, &tmax) || tmin > bestT) {
            continue;
        }

        let rv = node1.w;
        if (rv < 0.0) {
            // Leaf node: iterate triangles
            let start = i32(node0.w);
            let cnt   = -i32(rv);
            for (var j = 0; j < cnt; j = j + 1) {
                let triI = u32(start + j);
                let v0   = fetchtriPos(triI * 3u + 0u);
                let v1   = fetchtriPos(triI * 3u + 1u);
                let v2   = fetchtriPos(triI * 3u + 2u);
                let n0   = fetchtriNor(triI * 3u + 0u);
                let n1   = fetchtriNor(triI * 3u + 1u);
                let n2   = fetchtriNor(triI * 3u + 2u);

                // Back-face cull
                if (dot(normalize(cross(v1 - v0, v2 - v0)), rd) > 0.0) {
                    continue;
                }

                var t: f32; var u: f32; var v: f32;
                if (triIntersect(ro, rd, v0, v1, v2, &t, &u, &v) && t < bestT) {
                    bestT = t;
                    bestN = normalize(n0 * (1.0 - u - v) + n1 * u + n2 * v);
                }
            }
        } else {
            // Internal node: test both children and push near-first
            let c0  = i32(node0.w);
            let c1  = i32(rv);

            // Child 0 AABB
            let j0   = u32(c0 * 2);
            let mn0  = fetchbvhNodes(j0).xyz;
            let mx0  = fetchbvhNodes(j0 + 1u).xyz;
            // Child 1 AABB
            let j1   = u32(c1 * 2);
            let mn1  = fetchbvhNodes(j1).xyz;
            let mx1  = fetchbvhNodes(j1 + 1u).xyz;

            var t0min: f32; var t0max: f32;
            var t1min: f32; var t1max: f32;
            let hit0 = intersectAABB(ro, invRd, mn0, mx0, &t0min, &t0max) && t0min <= bestT;
            let hit1 = intersectAABB(ro, invRd, mn1, mx1, &t1min, &t1max) && t1min <= bestT;

            if (hit0 && hit1) {
                // Far-first push
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

    // 4) Shade if hit, otherwise sky
    if (bestT < 1e19) {
        let hit_os = ro + bestT * rd;
        let hit_ws = (uniforms.model * vec4<f32>(hit_os, 1.0)).xyz;

        // Transform normal from object→world
        if (length(bestN) < 1e-4 || !isSafeVec3(bestN)) {
            bestN = vec3<f32>(0.0, 1.0, 0.0);
        }
        var Nws = normalize(transpose(M3) * bestN);
        if (!isSafeVec3(Nws)) {
            Nws = vec3<f32>(0.0, 1.0, 0.0);
        }

        // Compute light direction
        var L = hit_ws - uniforms.lightPos.xyz;
        if (length(L) < 1e-4 || !isSafeVec3(L)) {
            L = vec3<f32>(0.0, 1.0, 0.0);
        }
        L = normalize(L);

        let diff = max(dot(Nws, L), 0.0);
        return vec4<f32>(diff, diff, diff, 1.0);
    }

    // Sky background
    return vec4<f32>(0.6, 0.8, 1.0, 1.0);
}
