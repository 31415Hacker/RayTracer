// morton_compute.wgsl

struct Bounds {
  min:       vec3<f32>,  // 12 bytes
  invExtent: vec3<f32>,  // 12 bytes
};

// binding(0): per-vertex position as vec4<f32> (x,y,z,0)
@group(0) @binding(0) var<storage, read>        positions   : array<vec4<f32>>;
// binding(1): flattened triangle indices (u32 × 3 per tri)
@group(0) @binding(1) var<storage, read>        indices     : array<u32>;
// binding(2): padded uniform struct (32 bytes)
@group(0) @binding(2) var<uniform>              bounds      : Bounds;
// binding(3): output Morton codes
@group(0) @binding(3) var<storage, read_write> mortonCodes : array<u32>;

// Spread 10-bit → 30-bit
fn expandBits(v: u32) -> u32 {
  var x = v & 0x3FFu;
  x = (x * 0x00010001u) & 0xFF0000FFu;
  x = (x * 0x00000101u) & 0x0F00F00Fu;
  x = (x * 0x00000011u) & 0xC30C30C3u;
  x = (x * 0x00000005u) & 0x49249249u;
  return x;
}

// Quantize normalized coord [0,1) → integer [0..1023]
fn quantize(x: f32) -> u32 {
  // floor(x*1024) yields 0..1023 for x in [0,1)
  return u32(clamp(floor(x * 1024.0), 0.0, 1023.0));
}

@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let tri = gid.x;
  // make sure we have 3 indices
  if (tri * 3u + 2u >= arrayLength(&indices)) {
    return;
  }

  // 1) load the 3 vertex indices
  let i0 = indices[tri*3u + 0u];
  let i1 = indices[tri*3u + 1u];
  let i2 = indices[tri*3u + 2u];

  // 2) fetch .xyz from the vec4 positions
  let p0 = positions[i0].xyz;
  let p1 = positions[i1].xyz;
  let p2 = positions[i2].xyz;

  // 3) compute centroid as vec3
  let ctr = (p0 + p1 + p2) / 3.0;

  // 4) normalize into [0,1) using your padded Bounds
  let nrm = (ctr - bounds.min) * bounds.invExtent;

  // 5) quantize each component to 10 bits
  let xi = quantize(nrm.x);
  let yi = quantize(nrm.y);
  let zi = quantize(nrm.z);

  // 6) expand & interleave bits
  let xx = expandBits(xi);
  let yy = expandBits(yi);
  let zz = expandBits(zi);

  // 7) write the 30-bit Morton code
  mortonCodes[tri] = (xx << 2u) | (yy << 1u) | zz;
}
