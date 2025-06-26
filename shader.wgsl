// ----------------------------------------
// Optimized shader.wgsl
// ----------------------------------------

// Uniforms (unchanged)
struct Uniforms {
    cameraPos:  vec4<f32>,
    lightPos:   vec4<f32>,
    camRight:   vec4<f32>,
    camUp:      vec4<f32>,
    camForward: vec4<f32>,
    model:      mat4x4<f32>,
};

@group(0) @binding(0) var<storage, read> triPos0:   array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> triPos1:   array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> triNor0:   array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> triNor1:   array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> bvhNodes0: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> bvhNodes1: array<vec4<f32>>;
@group(0) @binding(6) var<uniform>    uniforms:   Uniforms;

// Increased sample counts for higher quality
const SOFT_SAMPLES: u32 = 8u;
const GI_SAMPLES:   u32 = 16u;

// Array-split counts (injected by JS)
const COUNT_POS0: u32 = __COUNT_POS0__;
const COUNT_NOR0: u32 = __COUNT_NOR0__;
const COUNT_BVH0: u32 = __COUNT_BVH0__;

// Expanded Poisson-disk offsets for soft shadows (8 samples)
const POISSON_DISK: array<vec2<f32>, SOFT_SAMPLES> = array<vec2<f32>, SOFT_SAMPLES>(
    vec2<f32>(-0.7071,  0.7071),
    vec2<f32>( 0.7071,  0.7071),
    vec2<f32>( 0.7071, -0.7071),
    vec2<f32>(-0.7071, -0.7071),
    vec2<f32>(-0.3536,  0.9239),
    vec2<f32>( 0.9239,  0.3536),
    vec2<f32>( 0.3536, -0.9239),
    vec2<f32>(-0.9239, -0.3536)
);

// Expanded hemisphere samples for GI (16 samples - stratified)
const HEMI_SAMPLES: array<vec2<f32>, GI_SAMPLES> = array<vec2<f32>, GI_SAMPLES>(
    vec2<f32>(0.0625, 0.0625), vec2<f32>(0.1875, 0.0625), vec2<f32>(0.3125, 0.0625), vec2<f32>(0.4375, 0.0625),
    vec2<f32>(0.5625, 0.0625), vec2<f32>(0.6875, 0.0625), vec2<f32>(0.8125, 0.0625), vec2<f32>(0.9375, 0.0625),
    vec2<f32>(0.0625, 0.1875), vec2<f32>(0.1875, 0.1875), vec2<f32>(0.3125, 0.1875), vec2<f32>(0.4375, 0.1875),
    vec2<f32>(0.5625, 0.1875), vec2<f32>(0.6875, 0.1875), vec2<f32>(0.8125, 0.1875), vec2<f32>(0.9375, 0.1875)
);

// Cached BVH-node layout
struct CachedNode {
    bounds_min: vec3<f32>,
    bounds_max: vec3<f32>,
    child0:     i32,
    child1:     i32,
    is_leaf:    bool,
    tri_start:  i32,
    tri_count:  i32,
};

// Vertex→fragment data
struct VSOut {
    @builtin(position) Position: vec4<f32>,
    @location(0)       uv:       vec2<f32>,
};

@vertex
fn vs_main(@location(0) pos: vec2<f32>) -> VSOut {
    var o: VSOut;
    o.Position = vec4<f32>(pos, 0.0, 1.0);
    o.uv       = pos * vec2<f32>(0.5, 0.5) + vec2<f32>(0.5, 0.5);
    return o;
}

// Array-split fetches
fn fetchTriPos(idx: u32) -> vec3<f32> {
    if (idx < COUNT_POS0) {
        return triPos0[idx].xyz;
    }
    return triPos1[idx - COUNT_POS0].xyz;
}
fn fetchTriNor(idx: u32) -> vec3<f32> {
    if (idx < COUNT_NOR0) {
        return triNor0[idx].xyz;
    }
    return triNor1[idx - COUNT_NOR0].xyz;
}
fn fetchBVHNode(idx: u32) -> vec4<f32> {
    if (idx < COUNT_BVH0) {
        return bvhNodes0[idx];
    }
    return bvhNodes1[idx - COUNT_BVH0];
}

// Decode two vec4s into a CachedNode
fn fetchCachedNode(nodeIdx: i32) -> CachedNode {
    let base = u32(nodeIdx) * 2u;
    let b0   = fetchBVHNode(base);
    let b1   = fetchBVHNode(base + 1u);
    var n: CachedNode;
    n.bounds_min = b0.xyz;
    n.bounds_max = b1.xyz;
    n.is_leaf    = (b1.w < 0.0);
    if (n.is_leaf) {
        n.tri_start = i32(b0.w);
        n.tri_count = -i32(b1.w);
        n.child0    = -1;
        n.child1    = -1;
    } else {
        n.child0    = i32(b0.w);
        n.child1    = i32(b1.w);
        n.tri_start = -1;
        n.tri_count = 0;
    }
    return n;
}

// Fast AABB intersection
fn intersectAABB_fast(
    ro: vec3<f32>,
    invRd: vec3<f32>,
    mn: vec3<f32>,
    mx: vec3<f32>
) -> vec2<f32> {
    let t0    = (mn - ro) * invRd;
    let t1    = (mx - ro) * invRd;
    let tmin3 = min(t0, t1);
    let tmax3 = max(t0, t1);
    let tmin  = max(max(tmin3.x, tmin3.y), tmin3.z);
    let tmax  = min(min(tmax3.x, tmax3.y), tmax3.z);
    return vec2<f32>(tmin, tmax);
}

// Fast triangle-intersection (Möller–Trumbore)
fn triIntersect_fast(
    ro: vec3<f32>,
    rd: vec3<f32>,
    v0: vec3<f32>,
    v1: vec3<f32>,
    v2: vec3<f32>
) -> vec4<f32> {
    let e1  = v1 - v0;
    let e2  = v2 - v0;
    let p   = cross(rd, e2);
    let det = dot(e1, p);
    if (det < 1e-8) {
        return vec4<f32>(-1.0);
    }
    let invDet = 1.0 / det;
    let s      = ro - v0;
    let u      = dot(s, p) * invDet;
    if (u < 0.0 || u > 1.0) {
        return vec4<f32>(-1.0);
    }
    let q = cross(s, e1);
    let v = dot(rd, q) * invDet;
    if (v < 0.0 || u + v > 1.0) {
        return vec4<f32>(-1.0);
    }
    let t = dot(e2, q) * invDet;
    if (t <= 1e-6) {
        return vec4<f32>(-1.0);
    }
    return vec4<f32>(t, u, v, 1.0);
}

// Safe-vector check
fn isSafeVec(v: vec3<f32>) -> bool {
    return length(v) < 1e19 && all(abs(v) <= vec3<f32>(1e30));
}

// Generate camera ray
fn generateRay(uv: vec2<f32>) -> vec3<f32> {
    let scr = uv * 2.0 - vec2<f32>(1.0, 1.0);
    return normalize(
        uniforms.camRight.xyz   * scr.x +
        uniforms.camUp.xyz      * scr.y +
        uniforms.camForward.xyz
    );
}

// Hit record
struct Hit {
    t: f32,
    n: vec3<f32>,
};

// Optimized BVH traversal with node caching
const MAX_STACK: i32 = 16;
fn traverseBVH_optimized(
    ro: vec3<f32>,
    rd: vec3<f32>,
    invRd: vec3<f32>,
    maxDist: f32
) -> Hit {
    var stack: array<i32, MAX_STACK>;
    var sp:    i32 = 0;
    var cur:   i32 = 0;
    var bestT: f32 = maxDist;
    var bestN: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);

    loop {
        if (cur < 0 || sp >= MAX_STACK) {
            break;
        }
        let node = fetchCachedNode(cur);
        let hb   = intersectAABB_fast(ro, invRd, node.bounds_min, node.bounds_max);
        if (hb.y >= max(hb.x, 0.0) && hb.x <= bestT) {
            if (node.is_leaf) {
                // leaf: test triangles
                for (var j: i32 = 0; j < node.tri_count; j = j + 1) {
                    let ti = u32(node.tri_start + j);
                    let va = fetchTriPos(ti * 3u + 0u);
                    let vb = fetchTriPos(ti * 3u + 1u);
                    let vc = fetchTriPos(ti * 3u + 2u);
                    if (dot(cross(vb - va, vc - va), rd) > 0.0) {
                        continue;
                    }
                    let h = triIntersect_fast(ro, rd, va, vb, vc);
                    if (h.w > 0.0 && h.x < bestT) {
                        bestT = h.x;
                        let n0 = fetchTriNor(ti * 3u + 0u);
                        let n1 = fetchTriNor(ti * 3u + 1u);
                        let n2 = fetchTriNor(ti * 3u + 2u);
                        bestN = normalize(n0 * (1.0 - h.y - h.z) + n1 * h.y + n2 * h.z);
                    }
                }
                if (sp > 0) {
                    sp = sp - 1;
                    cur = stack[sp];
                } else {
                    break;
                }
            } else {
                // internal: test children
                let c0idx = node.child0;
                let c1idx = node.child1;
                let c0n   = fetchCachedNode(c0idx);
                let c1n   = fetchCachedNode(c1idx);
                let c0b   = intersectAABB_fast(ro, invRd, c0n.bounds_min, c0n.bounds_max);
                let c1b   = intersectAABB_fast(ro, invRd, c1n.bounds_min, c1n.bounds_max);
                let hit0  = c0b.y >= max(c0b.x, 0.0) && c0b.x <= bestT;
                let hit1  = c1b.y >= max(c1b.x, 0.0) && c1b.x <= bestT;
                if (hit0 && hit1) {
                    if (c0b.x <= c1b.x) {
                        if (sp < MAX_STACK - 1) {
                            stack[sp] = c1idx;
                            sp = sp + 1;
                        }
                        cur = c0idx;
                    } else {
                        if (sp < MAX_STACK - 1) {
                            stack[sp] = c0idx;
                            sp = sp + 1;
                        }
                        cur = c1idx;
                    }
                } else if (hit0) {
                    cur = c0idx;
                } else if (hit1) {
                    cur = c1idx;
                } else {
                    if (sp > 0) {
                        sp = sp - 1;
                        cur = stack[sp];
                    } else {
                        break;
                    }
                }
            }
        } else {
            if (sp > 0) {
                sp = sp - 1;
                cur = stack[sp];
            } else {
                break;
            }
        }
    }

    return Hit(bestT, bestN);
}

// Simple shadow test
fn shadowRay(ro: vec3<f32>, rd: vec3<f32>, maxDist: f32) -> bool {
    let invRd = 1.0 / rd;
    var cur: i32 = 0;
    loop {
        if (cur < 0) {
            break;
        }
        let node = fetchCachedNode(cur);
        let hb   = intersectAABB_fast(ro, invRd, node.bounds_min, node.bounds_max);
        if (hb.y >= max(hb.x, 0.0) && hb.x <= maxDist) {
            if (node.is_leaf) {
                for (var j: i32 = 0; j < node.tri_count; j = j + 1) {
                    let ti = u32(node.tri_start + j);
                    let va = fetchTriPos(ti * 3u + 0u);
                    let vb = fetchTriPos(ti * 3u + 1u);
                    let vc = fetchTriPos(ti * 3u + 2u);
                    let h  = triIntersect_fast(ro, rd, va, vb, vc);
                    if (h.w > 0.0 && h.x < maxDist) {
                        return true;
                    }
                }
                break;
            } else {
                cur = node.child0;
            }
        } else {
            break;
        }
    }
    return false;
}

// RNG helpers
fn hash_u32(x: u32) -> u32 {
    var v = x;
    v = ((v >> 16u) ^ v) * 0x45d9f3bu;
    v = ((v >> 16u) ^ v) * 0x45d9f3bu;
    v = (v >> 16u) ^ v;
    return v;
}
fn rand2(seed: vec2<f32>) -> f32 {
    let xi = bitcast<u32>(seed.x);
    let yi = bitcast<u32>(seed.y);
    return f32(hash_u32(xi ^ (yi << 16u))) * (1.0 / 4294967296.0);
}

// Hemisphere sampling
fn sampleHemisphere_fast(n: vec3<f32>, idx: u32) -> vec3<f32> {
    let sample = HEMI_SAMPLES[idx];
    let theta  = sample.x * 6.28318530718; // 2π
    let phi    = acos(sqrt(sample.y));
    let sinP   = sin(phi);
    let cosP   = cos(phi);
    let sinT   = sin(theta);
    let cosT   = cos(theta);

    let up = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(n.y) < 0.999);
    let T  = normalize(cross(up, n));
    let B  = cross(n, T);

    return normalize(
        T * (sinP * cosT) +
        n *  cosP     +
        B * (sinP * sinT)
    );
}

// Direct lighting with early shadow-out
fn computeDirectRadiance_OS_fast(
    hit_os: vec3<f32>,
    n_os:   vec3<f32>,
    light_os: vec3<f32>
) -> f32 {
    let L    = normalize(light_os - hit_os);
    let diff = max(dot(n_os, L), 0.0);
    if (diff <= 0.0) {
        return 0.0;
    }
    let radius: f32 = uniforms.lightPos.w;
    var occluded: u32 = 0u;

    let perp1 = normalize(cross(n_os, L));
    let perp2 = cross(L, perp1);

    for (var i: u32 = 0u; i < SOFT_SAMPLES; i = i + 1u) {
        let off       = POISSON_DISK[i] * radius;
        let samplePos = light_os + off.x * perp1 + off.y * perp2;
        let ro2       = hit_os + n_os * 0.001;
        let rd2       = normalize(samplePos - hit_os);
        let sd        = length(samplePos - hit_os);
        if (shadowRay(ro2, rd2, sd - 0.001)) {
            occluded = occluded + 1u;
        }
    }

    let shadowFactor = 1.0 - f32(occluded) / f32(SOFT_SAMPLES);
    return diff * shadowFactor;
}

// Main lighting
fn computeLighting_optimized(
    ro_w: vec3<f32>,
    rd_w: vec3<f32>,
    uv:   vec2<f32>
) -> vec4<f32> {
    // Transform to object space
    let ro4_os = uniforms.model * vec4<f32>(ro_w, 1.0);
    let ro_os  = ro4_os.xyz;
    let R3     = mat3x3<f32>(
        uniforms.model[0].xyz,
        uniforms.model[1].xyz,
        uniforms.model[2].xyz
    );
    let rd_os  = normalize(R3 * rd_w);
    let invRd  = 1.0 / rd_os;
    let light4 = uniforms.model * uniforms.lightPos;
    let light_os = light4.xyz;

    // Primary ray
    let hit0 = traverseBVH_optimized(ro_os, rd_os, invRd, 1e20);
    if (hit0.t >= 1e19) {
        return vec4<f32>(0.6, 0.8, 1.0, 1.0);
    }

    let hit_pos = ro_os + hit0.t * rd_os;
    var n_os    = hit0.n;
    if (!isSafeVec(n_os)) {
        n_os = vec3<f32>(0.0, 1.0, 0.0);
    }

    // Direct
    let direct: f32 = computeDirectRadiance_OS_fast(hit_pos, n_os, light_os);

    // One-bounce GI
    var indirect: f32 = 0.0;
    let giProb: f32 = 0.9;
    for (var i: u32 = 0u; i < GI_SAMPLES; i = i + 1u) {
        let seed = vec2<f32>(uv.x + f32(i) * 0.031, uv.y + f32(i) * 0.047);
        if (rand2(seed) > giProb) {
            continue;
        }
        let dir1   = sampleHemisphere_fast(n_os, i);
        let ro1    = hit_pos + n_os * 0.001;
        let invRd1 = 1.0 / dir1;
        let h1     = traverseBVH_optimized(ro1, dir1, invRd1, 50.0);
        if (h1.t < 50.0) {
            let p1    = ro1 + h1.t * dir1;
            var nn1   = h1.n;
            if (!isSafeVec(nn1)) {
                nn1 = vec3<f32>(0.0, 1.0, 0.0);
            }
            indirect = indirect + computeDirectRadiance_OS_fast(p1, nn1, light_os) / giProb;
        }
    }
    indirect = indirect / f32(GI_SAMPLES);

    // Combine & gamma
    let albedo = vec3<f32>(1.0, 0.84, 0.0);
    let color  = (direct + indirect * 0.6) * albedo;
    return vec4<f32>(pow(color, vec3<f32>(1.0 / 2.2)), 1.0);
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let ro: vec3<f32> = uniforms.cameraPos.xyz;
    let rd: vec3<f32> = generateRay(in.uv);
    return computeLighting_optimized(ro, rd, in.uv);
}
