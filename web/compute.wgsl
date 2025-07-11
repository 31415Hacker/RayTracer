// compute.wgsl

// — Uniform block (resolution as vec2<f32>) —
struct Uniforms {
    cameraPos:  vec4<f32>,
    lightPos:   vec4<f32>,
    camRight:   vec4<f32>,
    camUp:      vec4<f32>,
    camForward: vec4<f32>,
    model:      mat4x4<f32>,
    resolution: vec2<f32>,
    time:       f32,
};

@group(0) @binding(0) var<storage, read> triPos0:   array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> triPos1:   array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> triNor0:   array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> triNor1:   array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> bvhNodes0: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> bvhNodes1: array<vec4<f32>>;
@group(0) @binding(6) var<uniform>      uniforms:  Uniforms;
@group(0) @binding(7) var               outputTex: texture_storage_2d<rgba8unorm, write>;

// — Quality params —
const SOFT_SAMPLES: u32 = 4u;
const GI_SAMPLES:   u32 = 32u;
const GI_BOUNCES:   u32 = 3u;

// — Hit record —
struct Hit {
    t: f32,
    n: vec3<f32>,
};

// — Cached BVH node layout —
struct CachedNode {
    bounds_min: vec3<f32>,
    bounds_max: vec3<f32>,
    child0:     i32,
    child1:     i32,
    is_leaf:    bool,
    tri_start:  i32,
    tri_count:  i32,
};

// — Fetch triangle position —
fn fetchTriPos(idx: u32) -> vec3<f32> {
    let from0 = triPos0[idx].xyz;
    let from1 = triPos1[idx - __COUNT_POS0__].xyz;
    return select(from1, from0, idx < __COUNT_POS0__);
}

// — Fetch triangle normal —
fn fetchTriNor(idx: u32) -> vec3<f32> {
    let from0 = triNor0[idx].xyz;
    let from1 = triNor1[idx - __COUNT_NOR0__].xyz;
    return select(from1, from0, idx < __COUNT_NOR0__);
}

// — Fetch raw BVH node entry —
fn fetchBVHNode(idx: u32) -> vec4<f32> {
    let from0 = bvhNodes0[idx];
    let from1 = bvhNodes1[idx - __COUNT_BVH0__];
    return select(from1, from0, idx < __COUNT_BVH0__);
}

// — Decode two vec4s into a CachedNode —
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

// — Fast AABB intersection (slab method) —
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

// — Fast Möller–Trumbore intersection —
fn triIntersect_fast(
    ro: vec3<f32>,
    rd: vec3<f32>,
    v0: vec3<f32>,
    v1: vec3<f32>,
    v2: vec3<f32>
) -> vec4<f32> {
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let p = cross(rd, e2);
    let det = dot(e1, p);

    let s = ro - v0;
    let invDet = 1.0 / det;
    let u = dot(s, p) * invDet;

    let q = cross(s, e1);
    let v = dot(rd, q) * invDet;

    let t = dot(e2, q) * invDet;

    // Combine all validity conditions into one boolean
    let hit = ((((det > 1e-10) & (u >= 0.0)) & (v >= 0.0)) & (((u + v) <= 1.0) & (t > 1e-10)));

    return select(vec4<f32>(-1.0, 0.0, 0.0, 0.0), vec4<f32>(t, u, v, 1.0), hit);
}

// — PRNG helpers for deterministic pseudo‐randoms —
fn hash_u32(seed: u32) -> u32 {
    var x = seed;
    x = (x ^ 61u) ^ (x >> 16u);
    x = x + (x << 3u);
    x = x ^ (x >> 4u);
    x = x * 0x27d4eb2du;
    x = x ^ (x >> 15u);
    return x;
}
fn rand2(seed: vec2<f32>) -> f32 {
    let xi = u32((seed.x * 43758.0) + 24521); // Use large primes
    let yi = u32((seed.y * 199999.0) + 1423);
    let mix = xi ^ (yi >> 6u) ^ (yi << 11u);
    return f32(hash_u32(mix)) * (1.0 / 4294967296.0);
}

// — Importance-sampled hemisphere (cosine Lambertian) —
const TWO_PI: f32 = 6.283185307179586;
fn sampleHemisphere_importance(n: vec3<f32>, seed: vec2<f32>) -> vec3<f32> {
    let theta = TWO_PI * seed.x;
    let r     = sqrt(seed.y);
    let x     = r * cos(theta);
    let y     = r * sin(theta);
    let z     = sqrt(max(0.0, 1.0 - seed.y));
    let up    = select(vec3<f32>(1,0,0), vec3<f32>(0,1,0), abs(n.y) < 0.999);
    let T     = normalize(cross(up, n));
    let B     = cross(n, T);
    return normalize(T * x + B * y + n * z);
}

// — Shadow-ray test with early-out —
fn shadowRay(ro: vec3<f32>, rd: vec3<f32>, maxDist: f32) -> bool {
    let invRd = 1.0 / rd;
    var cur: i32 = 0;
    loop {
        if (cur < 0) { break; }
        let node = fetchCachedNode(cur);
        let hb   = intersectAABB_fast(ro, invRd, node.bounds_min, node.bounds_max);
        if ((hb.y >= max(hb.x, 0.0)) & (hb.x <= maxDist)) {
            if (node.is_leaf) {
                for (var j: i32 = 0; j < node.tri_count; j = j + 1) {
                    let ti = u32(node.tri_start + j);
                    let va = fetchTriPos(ti*3u + 0u);
                    let vb = fetchTriPos(ti*3u + 1u);
                    let vc = fetchTriPos(ti*3u + 2u);
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

// — Direct lighting with soft shadows —
fn computeDirectRadiance_OS_fast(
    hit_os:   vec3<f32>,
    n_os:     vec3<f32>,
    light_os: vec3<f32>,
    seed_id:  u32  // <-- unique per pixel or sample
) -> f32 {
    let hitToLight = light_os - hit_os;
    let L    = normalize(hitToLight);
    let diff = max(dot(n_os, L), 0.0);
    if (diff <= 0.0) { return 0.0; }
    var occ: u32 = 0u;

    // Build orthonormal basis from surface normal
    let up        = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n_os.y) > 0.99);
    let tangent   = normalize(cross(up, n_os));
    let bitangent = cross(n_os, tangent);

    let ro2 = fma(n_os, vec3(0.001), hit_os);
    let lightDist  = length(hitToLight);

    for (var i: u32 = 0u; i < SOFT_SAMPLES; i = i + 1u) {
        // General stratified sampling for arbitrary GI_SAMPLES
        let sqrtS = ceil(sqrt(f32(GI_SAMPLES)));
        let cols  = u32(sqrtS);            // width of grid
        let rows  = (GI_SAMPLES + cols - 1u) / cols; // ceiling division for full grid coverage

        let sx = i % cols;
        let sy = i / cols;

        // Per-pixel jitter seeds
        let jitter1 = rand2(vec2<f32>(f32(seed_id + 17u) + f32(hash_u32(u32(uniforms.time))), f32(i) + 0.37));
        let jitter2 = rand2(vec2<f32>(f32(seed_id + 91u) + f32(hash_u32(u32(uniforms.time))), f32(i) + 1.13));

        let s1 = (f32(sx) + jitter1) / f32(cols);  // x stratified
        let s2 = (f32(sy) + jitter2) / f32(rows);  // y stratified

        // Cosine-weighted hemisphere importance sampling
        let theta = TWO_PI * s1;
        let r     = sqrt(s2);
        let x     = r * cos(theta);
        let y     = r * sin(theta);
        let z     = sqrt(max(0.0, 1.0 - s2));

        let dir_local = vec3<f32>(x, y, z);
        let dir_world = normalize(
            tangent   * dir_local.x +
            bitangent * dir_local.y +
            n_os      * dir_local.z
        );

        let rd2 = dir_world;

        let visible = dot(rd2, normalize(hitToLight)) > 0.5;
        let hit     = shadowRay(ro2, rd2, lightDist - 0.001);
        occ += select(0u, 1u, visible & hit);
    }

    let shadowFactor = fma(-f32(occ), 1.0 / f32(SOFT_SAMPLES), 1.0);;
    return diff * shadowFactor;
}

// — BVH traversal with fixed-size stack —
const MAX_STACK: i32 = 12;
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
    var bestN: vec3<f32> = vec3<f32>(0.0);
    loop {
        if (cur < 0 || sp >= MAX_STACK) { break; }
        let node = fetchCachedNode(cur);
        let hb   = intersectAABB_fast(ro, invRd, node.bounds_min, node.bounds_max);
        if (hb.y >= max(hb.x, 0.0) && hb.x <= bestT) {
            if (node.is_leaf) {
                for (var j: i32 = 0; j < node.tri_count; j = j + 1) {
                    let ti = u32(node.tri_start + j);
                    let va = fetchTriPos(ti*3u+0u);
                    let vb = fetchTriPos(ti*3u+1u);
                    let vc = fetchTriPos(ti*3u+2u);
                    if (dot(cross(vb - va, vc - va), rd) > 0.0) {
                        continue;
                    }
                    let h = triIntersect_fast(ro, rd, va, vb, vc);
                    if (h.w > 0.0 && h.x < bestT) {
                        bestT = h.x;
                        let n0 = fetchTriNor(ti*3u+0u);
                        let n1 = fetchTriNor(ti*3u+1u);
                        let n2 = fetchTriNor(ti*3u+2u);
                        bestN = normalize(fma(n2, vec3(h.z), fma(n1, vec3(h.y), n0 * (1.0 - h.y - h.z))));
                    }
                }
                if (sp > 0) {
                    sp = sp - 1;
                    cur = stack[sp];
                } else {
                    break;
                }
            } else {
                let c0 = node.child0;
                let c1 = node.child1;
                let n0 = fetchCachedNode(c0);
                let n1 = fetchCachedNode(c1);
                let b0 = intersectAABB_fast(ro, invRd, n0.bounds_min, n0.bounds_max);
                let b1 = intersectAABB_fast(ro, invRd, n1.bounds_min, n1.bounds_max);
                let hit0 = (b0.y >= max(b0.x,0.0) && b0.x <= bestT);
                let hit1 = (b1.y >= max(b1.x,0.0) && b1.x <= bestT);
                if (hit0 && hit1) {
                    if (b0.x <= b1.x) {
                        if (sp < MAX_STACK-1) {
                            stack[sp] = c1;
                            sp = sp + 1;
                        }
                        cur = c0;
                    } else {
                        if (sp < MAX_STACK-1) {
                            stack[sp] = c0;
                            sp = sp + 1;
                        }
                        cur = c1;
                    }
                } else if (hit0) {
                    cur = c0;
                } else if (hit1) {
                    cur = c1;
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

// — Full lighting with importance-sampled GI —
fn computeLighting_optimized(ro_w: vec3<f32>, rd_w: vec3<f32>, uv: vec2<f32>, pixelIndex: u32, time: f32) -> vec4<f32> {
    // Transform to object space
    let ro4    = uniforms.model * vec4<f32>(ro_w,1.0);
    var ro_os  = ro4.xyz;
    let R3     = mat3x3<f32>(
        uniforms.model[0].xyz,
        uniforms.model[1].xyz,
        uniforms.model[2].xyz
    );
    var rd_os  = normalize(R3 * rd_w);
    let invRd  = 1.0 / rd_os;
    let light4 = uniforms.model * uniforms.lightPos;
    let light_os = light4.xyz;

    // Primary ray
    let hit0 = traverseBVH_optimized(ro_os, rd_os, invRd, 1e20);
    if (hit0.t >= 1e19) {
        return vec4<f32>(0.0,0.0,0.0,1.0);
    }

    // Hit point & normal
    let hit_pos = fma(rd_os, vec3(hit0.t), ro_os);
    var n_os    = hit0.n;
    let albedo  = vec3<f32>(1.0,0.84,0.0);

    // Indirect GI
    var indirect: vec3<f32> = vec3<f32>(0.0);
    for (var i: u32 = 0u; i < GI_SAMPLES; i = i + 1u) {
        // generate two seeds per sample
        var throughput = albedo;
        var ro_bounce  = fma(n_os, vec3(0.001), hit_pos);
        var n_bounce   = n_os;
        for (var b: u32 = 0u; b < GI_BOUNCES; b = b + 1u) {
            let bounceSeed = f32(b) * 101.0 + f32(i);
            let pixelSeed = f32(pixelIndex) / f32(uniforms.resolution.x * uniforms.resolution.y);
            let s1 = rand2(vec2<f32>(
                bounceSeed,
                fma(pixelSeed, f32(i), 17.0) + time
            ));

            let s2 = rand2(vec2<f32>(
                fma(pixelSeed, f32(i), 37.0) + time,
                bounceSeed
            ));
            let dir = sampleHemisphere_importance(n_bounce, vec2<f32>(s1, s2));
            let hb  = traverseBVH_optimized(ro_bounce, dir, 1.0/dir, 1e20);
            if (hb.t >= 1e19) { break; }
            let pB = fma(dir, vec3(hb.t), ro_bounce);
            let nB = hb.n;
            let dS = computeDirectRadiance_OS_fast(pB, nB, light_os, pixelIndex);
            indirect += throughput * dS;
            throughput *= albedo;
            ro_bounce = fma(nB, vec3(0.0001), pB);
            n_bounce  = nB;
        }
    }
    indirect = indirect / f32(GI_SAMPLES);

    // Combine & gamma
    let col = fma(albedo, vec3(computeDirectRadiance_OS_fast(hit_pos, n_os, light_os, pixelIndex)), indirect);
    return vec4<f32>(pow(col, vec3<f32>(1.0/2.2)), 1.0);
}

// — Compute entrypoint (16×16×1) —
@compute @workgroup_size(16, 16, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = u32(uniforms.resolution.x);
    let h = u32(uniforms.resolution.y);
    if (gid.x >= w || gid.y >= h) { return; }

    let uv = vec2<f32>(
        (f32(gid.x) + 0.5) / f32(w),
        (f32(gid.y) + 0.5) / f32(h)
    );

    let ro = uniforms.cameraPos.xyz;
    let sx = fma(uv.x, 2.0, -1.0);
    let sy = fma(uv.y, 2.0, -1.0);

    let rd = normalize(
        fma(uniforms.camRight.xyz, vec3(sx),
            fma(uniforms.camUp.xyz, vec3(sy),
                uniforms.camForward.xyz))
    );

    let px = clamp(uv.x * f32(w), 0.0, f32(w - 1));
    let py = clamp(uv.y * f32(h), 0.0, f32(h - 1));

    // Convert to integers
    let ix = u32(px);
    let iy = u32(py);

    // Flatten 2D → 1D
    let pixelIndex = iy * w + ix;

    let col = computeLighting_optimized(ro, rd, uv, pixelIndex, f32(hash_u32(u32(uniforms.time))));
    textureStore(outputTex, vec2<i32>(i32(gid.x), i32(gid.y)), col);
}