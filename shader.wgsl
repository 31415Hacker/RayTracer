// shader.wgsl

//////////////////////////////////////////////////////////
// Camera & Buffer Bindings
//////////////////////////////////////////////////////////

struct Camera {
  origin:     vec4<f32>,    // xyz: camera position
  forward:    vec4<f32>,    // camera forward vector
  right:      vec4<f32>,    // camera right vector
  up:         vec4<f32>,    // camera up vector
  fl_aspect:  vec2<f32>,    // [focal_length, aspect_ratio]
  resolution: vec2<f32>,    // [width, height]
  model:      mat4x4<f32>   // model matrix (rotation only)
};

@group(0) @binding(0) var<uniform> cam      : Camera;
@group(0) @binding(1) var<storage,read> posBuf   : array<vec3<f32>>;
@group(0) @binding(2) var<storage,read> normBuf  : array<vec3<f32>>;
@group(0) @binding(3) var<storage,read> nodeBuf  : array<vec4<f32>>;

//////////////////////////////////////////////////////////
// Intersection Helpers
//////////////////////////////////////////////////////////

// Ray-AABB intersection -> [tmin, tmax]
fn intersectAABB(ro:vec3<f32>, invRd:vec3<f32>, mn:vec3<f32>, mx:vec3<f32>)
                 -> vec2<f32> {
  let t0 = (mn - ro) * invRd;
  let t1 = (mx - ro) * invRd;
  let tmin3 = min(t0, t1);
  let tmax3 = max(t0, t1);
  let tmin = max(max(tmin3.x, tmin3.y), tmin3.z);
  let tmax = min(min(tmax3.x, tmax3.y), tmax3.z);
  return vec2<f32>(tmin, tmax);
}

// Watertight, winding-agnostic ray–triangle test
// returns (t, u, v, 1.0) on hit, or (0,0,0,0) on miss
fn triIntersect(
    ro: vec3<f32>,
    rd: vec3<f32>,
    v0: vec3<f32>,
    v1: vec3<f32>,
    v2: vec3<f32>
) -> vec4<f32> {
  // 1) pick the projection axis
  let ax = abs(rd.x);
  let ay = abs(rd.y);
  let az = abs(rd.z);
  // Determine the dominant axis (kz) to project the triangle onto
  let kz = select(select(0u, 1u, ay > ax), 2u, az > max(ax, ay));
  let kx = (kz + 1u) % 3u;
  let ky = (kx + 1u) % 3u;

  // 2) shear & scale factors
  let d  = rd[kz];
  // Handle cases where d (ray component along kz) is zero or very small
  if (abs(d) < 1e-8) { return vec4<f32>(0); } // Ray is parallel to projection plane
  let Sx = rd[kx] / d;
  let Sy = rd[ky] / d;
  let Sz = 1.0 / d; // This is the inverse of d

  // 3) origin offset
  let Ok = ro[kz];
  let Ox = ro[kx];
  let Oy = ro[ky];

  // 4) shear all vertices
  let tz0 = v0[kz] - Ok;
  let tx0 = v0[kx] - Ox - Sx * tz0;
  let ty0 = v0[ky] - Oy - Sy * tz0;
  let v0s = vec3<f32>(tx0, ty0, tz0 * Sz); // tz0 * Sz effectively scales tz0 by 1/d

  let tz1 = v1[kz] - Ok;
  let tx1 = v1[kx] - Ox - Sx * tz1;
  let ty1 = v1[ky] - Oy - Sy * tz1;
  let v1s = vec3<f32>(tx1, ty1, tz1 * Sz);

  let tz2 = v2[kz] - Ok;
  let tx2 = v2[kx] - Ox - Sx * tz2;
  let ty2 = v2[ky] - Oy - Sy * tz2;
  let v2s = vec3<f32>(tx2, ty2, tz2 * Sz);

  // 5) 2D edge-functions (cross products in 2D)
  let e0 = v1s.x * v2s.y - v1s.y * v2s.x;
  let e1 = v2s.x * v0s.y - v2s.y * v0s.x;
  let e2 = v0s.x * v1s.y - v0s.y * v1s.x;

  // 6) reject if signs are mixed (point is outside triangle in 2D projection)
  let hasNeg = (e0 < 0.0) || (e1 < 0.0) || (e2 < 0.0);
  let hasPos = (e0 > 0.0) || (e1 > 0.0) || (e2 > 0.0);
  if (hasNeg && hasPos) {
    return vec4<f32>(0); // Ray misses the triangle in 2D
  }

  // 7) compute t (distance along ray)
  let sum = e0 + e1 + e2;
  // If sum is zero, the triangle is degenerate or parallel to the projection plane
  if (abs(sum) < 1e-8) { // Use an epsilon for float comparison
    return vec4<f32>(0);
  }
  let tScaled = e0 * v0s.z + e1 * v1s.z + e2 * v2s.z;
  let t = tScaled / sum; // This is the actual t value along the ray

  // Reject hits behind the ray origin
  if (t <= 0.0) { // Or a small epsilon like 1e-4 to avoid self-intersection issues
    return vec4<f32>(0);
  }

  // 8) barycentrics
  let invSum = 1.0 / sum;
  let u = e0 * invSum;
  let v = e1 * invSum;

  return vec4<f32>(t, u, v, 1.0); // Return hit (t, u, v, 1.0)
}

//////////////////////////////////////////////////////////
// Full-Screen Quad Vertex Shader
//////////////////////////////////////////////////////////

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
  // A standard full-screen triangle-strip quad
  var quad = array<vec2<f32>,4>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0, -1.0),
    vec2<f32>(-1.0,  1.0),
    vec2<f32>( 1.0,  1.0)
  );
  return vec4<f32>(quad[vi], 0.0, 1.0);
}

//////////////////////////////////////////////////////////
// Ray-Tracing Fragment Shader - FIXED RENDERING BUG
//////////////////////////////////////////////////////////

@fragment
fn fs_main(@builtin(position) fc: vec4<f32>) -> @location(0) vec4<f32> {
  // 1) Compute NDC uv in [-1,1]
  let uv = (fc.xy / cam.resolution) * 2.0 - vec2<f32>(1.0, 1.0);

  // 2) Build world-space ray
  let fl   = cam.fl_aspect.x;
  let asp  = cam.fl_aspect.y;
  let ro_w = cam.origin.xyz;
  let rd_w = normalize(
    cam.forward.xyz * fl +
    cam.right.xyz   * uv.x * asp +
    cam.up.xyz      * uv.y
  );

  // 3) Transform ray into object space (rotation only)
  let R    = mat3x3<f32>(
    cam.model[0].xyz,
    cam.model[1].xyz,
    cam.model[2].xyz
  );
  let invM = transpose(R);
  let ro    = invM * ro_w;
  let rd    = invM * rd_w;
  let invRd = vec3<f32>(
    select(1.0/rd.x, 1e20, abs(rd.x) < 1e-8),
    select(1.0/rd.y, 1e20, abs(rd.y) < 1e-8),
    select(1.0/rd.z, 1e20, abs(rd.z) < 1e-8)
  );

  // 4) BVH traversal - FIXED TRIANGLE INDEXING
  var stack: array<u32, 64>;
  var sp = 1u;
  stack[0] = 0u;
  var bestT = 1e20;
  var bestN = vec3<f32>(0.0);
  let MAX_STACK = 64u;

  loop {
    if (sp == 0u) { break; }
    sp -= 1u;
    let ni = stack[sp];

    let d0 = nodeBuf[ni*2u + 0u];
    let d1 = nodeBuf[ni*2u + 1u];
    let mn = d0.xyz;
    let mx = d1.xyz;
    let leftOrStart = i32(round(d0.w));
    let rightOrCount = i32(round(d1.w));

    let ts = intersectAABB(ro, invRd, mn, mx);
    if (ts.x > ts.y || ts.x > bestT) {
      continue;
    }

    if (rightOrCount < 0) {
      // FIX: Correct triangle indexing - use base triangle index directly
      let startTri = u32(leftOrStart);  // Starting triangle index
      let cnt   = u32(-rightOrCount);   // Triangle count
      
      for (var j = 0u; j < cnt; j++) {
        let triIdx = startTri + j;      // Current triangle index
        let baseIdx = triIdx * 3u;      // Base vertex index for this triangle
        
        // Fetch triangle vertices - FIXED INDEXING
        let v0 = posBuf[baseIdx + 0u];
        let v1 = posBuf[baseIdx + 1u];
        let v2 = posBuf[baseIdx + 2u];
        
        let ri = triIntersect(ro, rd, v0, v1, v2);

        if (ri.w > 0.5 && ri.x < bestT) {
          bestT = ri.x;
          
          // Fetch corresponding normals
          let n0 = normBuf[baseIdx + 0u];
          let n1 = normBuf[baseIdx + 1u];
          let n2 = normBuf[baseIdx + 2u];
          
          // Interpolate normal using barycentric coordinates
          bestN = normalize(n0*(1.0-ri.y-ri.z) + n1*ri.y + n2*ri.z);
        }
      }
    } else {
      // Internal node
      let c0 = u32(leftOrStart);
      let c1 = u32(rightOrCount);

      let ts0 = intersectAABB(ro, invRd, 
                 nodeBuf[c0*2u].xyz, nodeBuf[c0*2u+1u].xyz);
      let ts1 = intersectAABB(ro, invRd,
                 nodeBuf[c1*2u].xyz, nodeBuf[c1*2u+1u].xyz);

      let hit0 = ts0.x <= ts0.y && ts0.x <= bestT;
      let hit1 = ts1.x <= ts1.y && ts1.x <= bestT;

      if (hit0 && hit1) {
        if (ts0.x < ts1.x) {
          if (sp < MAX_STACK-1u) {
            stack[sp] = c1; sp += 1u;
            stack[sp] = c0; sp += 1u;
          }
        } else {
          if (sp < MAX_STACK-1u) {
            stack[sp] = c0; sp += 1u;
            stack[sp] = c1; sp += 1u;
          }
        }
      } else if (hit0 && sp < MAX_STACK) {
        stack[sp] = c0; sp += 1u;
      } else if (hit1 && sp < MAX_STACK) {
        stack[sp] = c1; sp += 1u;
      }
    }
  }

  // 5) Shading
  if (bestT < 1e19) {
    let hit = ro + rd * bestT;
    let lightPos_world = vec3<f32>(5.0, 5.0, 5.0);
    let lightPos = invM * lightPos_world;
    
    let L = normalize(lightPos - hit);
    let V = normalize(-rd);
    let H = normalize(L + V);
    let diff = max(dot(bestN, L), 0.0);
    let spec = pow(max(dot(bestN, H), 0.0), 32.0);
    let color = vec3<f32>(1.0) * (0.8 * diff + 0.4 * spec);
    return vec4<f32>(color, 1.0);
  }

  // Background
  return vec4<f32>(0.6, 0.8, 1.0, 1.0);
}