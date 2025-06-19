// — Compute shader: generate (mortonCode, triIndex) pairs —
//    centroids: array<vec4<f32>> normalized to [0,1) in .xyz
//    mortonPairs: array<vec4<u32>> output, storing (code, index, 0,0)

@group(0) @binding(0) var<storage, read> centroids: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> mortonPairs: array<vec4<u32>>;

// Expand 10-bit to interleaved Morton bits
fn expandBits(v: u32) -> u32 {
    var x = v;
    x = (x * 0x00010001u) & 0xFF0000FFu;
    x = (x * 0x00000101u) & 0x0F00F00Fu;
    x = (x * 0x00000011u) & 0xC30C30C3u;
    x = (x * 0x00000005u) & 0x49249249u;
    return x;
}

fn morton3(x: f32, y: f32, z: f32) -> u32 {
    let cx = clamp(x, 0.0, 0.999999);
    let cy = clamp(y, 0.0, 0.999999);
    let cz = clamp(z, 0.0, 0.999999);
    let xi = u32(cx * 1024.0);
    let yi = u32(cy * 1024.0);
    let zi = u32(cz * 1024.0);
    return (expandBits(xi) << 2) | (expandBits(yi) << 1) | expandBits(zi);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&centroids)) { return; }
    let c = centroids[i].xyz;
    let code = morton3(c.x, c.y, c.z);
    mortonPairs[i] = vec4<u32>(code, i, 0u, 0u);
}