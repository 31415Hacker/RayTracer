// script.js

// — Shared state (so updateCamera/onMouseMove see them) —
let cameraPos = [0, 0, 5];
let yaw = 0;
let pitch = 0;
let keys = [];

// — Global GPU objects —
let device, context;
let uniformBuf;
let pos0Buf, pos1Buf, nor0Buf, nor1Buf, bvh0Buf, bvh1Buf;
let computePipeline, computeBindGroup;
let blitPipeline, blitBindGroup;
let quadBuf, sampler, outputTex;

// — Helpers —
const sliders = {
  scaleX: document.getElementById("scaleX"),
  scaleY: document.getElementById("scaleY"),
  scaleZ: document.getElementById("scaleZ"),
  rotX: document.getElementById("rotX"),
  rotY: document.getElementById("rotY"),
  rotZ: document.getElementById("rotZ"),
  posX: document.getElementById("posX"),
  posY: document.getElementById("posY"),
  posZ: document.getElementById("posZ"),
};

function makeTRS(rot, scale, pos) {
  const [sx, sy, sz] = scale;
  const [rx, ry, rz] = rot;
  const [tx, ty, tz] = pos;

  const cos = Math.cos;
  const sin = Math.sin;

  const cx = cos(rx),
    sx_ = sin(rx);
  const cy = cos(ry),
    sy_ = sin(ry);
  const cz = cos(rz),
    sz_ = sin(rz);

  // Rotation ZYX order
  const m00 = cz * cy;
  const m01 = cz * sy_ * sx_ - sz_ * cx;
  const m02 = cz * sy_ * cx + sz_ * sx_;
  const m10 = sz_ * cy;
  const m11 = sz_ * sy_ * sx_ + cz * cx;
  const m12 = sz_ * sy_ * cx - cz * sx_;
  const m20 = -sy_;
  const m21 = cy * sx_;
  const m22 = cy * cx;

  // Apply scaling
  return new Float32Array([
    m00 * sx,
    m01 * sy,
    m02 * sz,
    0,
    m10 * sx,
    m11 * sy,
    m12 * sz,
    0,
    m20 * sx,
    m21 * sy,
    m22 * sz,
    0,
    tx,
    ty,
    tz,
    1,
  ]);
}

function invert4(out, m) {
  const a00 = m[0],
    a01 = m[1],
    a02 = m[2],
    a03 = m[3];
  const a10 = m[4],
    a11 = m[5],
    a12 = m[6],
    a13 = m[7];
  const a20 = m[8],
    a21 = m[9],
    a22 = m[10],
    a23 = m[11];
  const a30 = m[12],
    a31 = m[13],
    a32 = m[14],
    a33 = m[15];

  const b00 = a00 * a11 - a01 * a10;
  const b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11;
  const b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30;
  const b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31;
  const b11 = a22 * a33 - a23 * a32;

  // determinant
  let det =
    b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;

  if (!det) return false;
  det = 1.0 / det;

  out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
  out[1] = (-a01 * b11 + a02 * b10 - a03 * b09) * det;
  out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
  out[3] = (-a21 * b05 + a22 * b04 - a23 * b03) * det;
  out[4] = (-a10 * b11 + a12 * b08 - a13 * b07) * det;
  out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
  out[6] = (-a30 * b05 + a32 * b02 - a33 * b01) * det;
  out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
  out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
  out[9] = (-a00 * b10 + a01 * b08 - a03 * b06) * det;
  out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
  out[11] = (-a20 * b04 + a21 * b02 - a23 * b00) * det;
  out[12] = (-a10 * b09 + a11 * b07 - a12 * b06) * det;
  out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
  out[14] = (-a30 * b03 + a31 * b01 - a32 * b00) * det;
  out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;

  return true;
}

// Update and return 4×4 TRS matrix as Float32Array
function updateTRSMatrix() {
  const scale = [
    parseFloat(sliders.scaleX.value),
    parseFloat(sliders.scaleY.value),
    parseFloat(sliders.scaleZ.value),
  ];
  const pos = [
    parseFloat(sliders.posX.value),
    parseFloat(sliders.posY.value),
    parseFloat(sliders.posZ.value),
  ];
  const rot = [
    parseFloat((sliders.rotX.value * Math.PI) / 180) - Math.PI,
    parseFloat((sliders.rotY.value * Math.PI) / 180),
    parseFloat((sliders.rotZ.value * Math.PI) / 180),
  ];

  return makeTRS(rot, scale, pos);
}

// Load text (WGSL or any) over HTTP
async function loadText(url) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`Failed to load ${url}`);
  return r.text();
}

// Load a simple Wavefront OBJ (positions, normals, faces)
async function loadObj(url) {
  const txt = await loadText(url);
  const pos = [],
    nor = [],
    vIdx = [],
    nIdx = [];
  function toIndex(raw, count) {
    const i = parseInt(raw, 10);
    return i > 0 ? i - 1 : count + i;
  }
  for (let line of txt.split("\n")) {
    line = line.trim();
    if (line.startsWith("v ")) {
      const [, x, y, z] = line.split(/\s+/);
      pos.push(+x, +y, +z);
    } else if (line.startsWith("vn ")) {
      const [, x, y, z] = line.split(/\s+/);
      nor.push(+x, +y, +z);
    } else if (line.startsWith("f ")) {
      const refs = line
        .split(/\s+/)
        .slice(1)
        .map((p) => p.split("/"));
      const vC = pos.length / 3,
        nC = nor.length / 3;
      for (let i = 1; i < refs.length - 1; i++) {
        const [v0, , n0] = refs[0],
          [v1, , n1] = refs[i],
          [v2, , n2] = refs[i + 1];
        vIdx.push(toIndex(v0, vC), toIndex(v1, vC), toIndex(v2, vC));
        nIdx.push(toIndex(n0, nC), toIndex(n1, nC), toIndex(n2, nC));
      }
    }
  }
  return {
    positions: new Float32Array(pos),
    normals: new Float32Array(nor),
    vertIdx: new Uint32Array(vIdx),
    normIdx: new Uint32Array(nIdx),
  };
}

// Normalize a mesh into [-1,1]^3
function normalizeMesh(m) {
  const p = m.positions,
    n = p.length / 3;
  const mn = [Infinity, Infinity, Infinity],
    mx = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 3; k++) {
      const v = p[3 * i + k];
      mn[k] = Math.min(mn[k], v);
      mx[k] = Math.max(mx[k], v);
    }
  }
  const ctr = mn.map((v, i) => (v + mx[i]) * 0.5);
  const sc = 1 / Math.max(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 3; k++) {
      p[3 * i + k] = (p[3 * i + k] - ctr[k]) * sc;
    }
  }
  return m;
}

// — Morton compute pipeline setup —
let mortonPipeline, mortonBindGroupLayout;
async function initMortonPipeline() {
  const url = `morton_compute.wgsl?v=${Date.now()}`;
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`Failed to load ${url}`);
  const code = await r.text();
  console.log(
    "--- loaded Morton shader ---\n",
    code,
    "\n----------------------------"
  );
  const module = device.createShaderModule({ code });
  mortonPipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "main" },
    label: "MortonPipeline",
  });
  mortonBindGroupLayout = mortonPipeline.getBindGroupLayout(0);
}

// — Helper: upload TypedArray as STORAGE buffer —
function createStorageBufferFromArray(arr) {
  const buf = device.createBuffer({
    size: arr.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buf, 0, arr.buffer, arr.byteOffset, arr.byteLength);
  return buf;
}

// — Helper: read back GPU buffer to ArrayBuffer —
async function readbackBuffer(srcBuf, size) {
  const dst = device.createBuffer({
    size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(srcBuf, 0, dst, 0, size);
  device.queue.submit([enc.finish()]);
  await dst.mapAsync(GPUMapMode.READ);
  const copy = dst.getMappedRange().slice(0);
  dst.unmap();
  dst.destroy();
  return copy;
}

/**
 * Build a SAH BVH on the CPU, using true triangle AABBs.
 *
 * @param mesh  An object with:
 *   - positions: Float32Array of length V*3
 *   - vertIdx:   Uint32Array   of length T*3
 * @param leaf  Maximum triangles per leaf
 * @returns { nodes: Array, triOrder: Uint32Array }
 */
function buildLBVH(mesh, leaf = 1) {
  const tris = mesh.vertIdx.length / 3;

  // --- 1) Precompute per-triangle AABB and centroid ---
  const triMin = new Float32Array(tris * 3);
  const triMax = new Float32Array(tris * 3);
  const centroids = new Float32Array(tris * 3);

  for (let t = 0; t < tris; ++t) {
    const i0 = mesh.vertIdx[3 * t + 0] * 3;
    const i1 = mesh.vertIdx[3 * t + 1] * 3;
    const i2 = mesh.vertIdx[3 * t + 2] * 3;

    // vertex positions
    const x0 = mesh.positions[i0],
      y0 = mesh.positions[i0 + 1],
      z0 = mesh.positions[i0 + 2];
    const x1 = mesh.positions[i1],
      y1 = mesh.positions[i1 + 1],
      z1 = mesh.positions[i1 + 2];
    const x2 = mesh.positions[i2],
      y2 = mesh.positions[i2 + 1],
      z2 = mesh.positions[i2 + 2];

    // AABB
    triMin[3 * t + 0] = Math.min(x0, x1, x2);
    triMin[3 * t + 1] = Math.min(y0, y1, y2);
    triMin[3 * t + 2] = Math.min(z0, z1, z2);

    triMax[3 * t + 0] = Math.max(x0, x1, x2);
    triMax[3 * t + 1] = Math.max(y0, y1, y2);
    triMax[3 * t + 2] = Math.max(z0, z1, z2);

    // centroid
    centroids[3 * t + 0] = (x0 + x1 + x2) / 3;
    centroids[3 * t + 1] = (y0 + y1 + y2) / 3;
    centroids[3 * t + 2] = (z0 + z1 + z2) / 3;
  }

  // --- 2) Initial triangle order and node array ---
  const triOrder = new Uint32Array(tris);
  for (let i = 0; i < tris; ++i) triOrder[i] = i;
  const nodes = [];

  // --- 3) Recursive SAH split ---
  function recurse(start, end) {
    const nodeIdx = nodes.length;
    nodes.push(null);
    const count = end - start;

    // Compute this node's *true* AABB from triMin/triMax
    const mn = [Infinity, Infinity, Infinity];
    const mx = [-Infinity, -Infinity, -Infinity];
    for (let i = start; i < end; ++i) {
      const t = triOrder[i];
      for (let k = 0; k < 3; ++k) {
        mn[k] = Math.min(mn[k], triMin[3 * t + k]);
        mx[k] = Math.max(mx[k], triMax[3 * t + k]);
      }
    }

    // Leaf?
    if (count <= leaf) {
      nodes[nodeIdx] = { mn, mx, left: start, right: -count };
      return nodeIdx;
    }

    // Compute centroid bounds to pick split axis
    const cMn = [Infinity, Infinity, Infinity];
    const cMx = [-Infinity, -Infinity, -Infinity];
    for (let i = start; i < end; ++i) {
      const off = 3 * triOrder[i];
      for (let k = 0; k < 3; ++k) {
        const v = centroids[off + k];
        cMn[k] = Math.min(cMn[k], v);
        cMx[k] = Math.max(cMx[k], v);
      }
    }
    const ext = [cMx[0] - cMn[0], cMx[1] - cMn[1], cMx[2] - cMn[2]];
    let axis =
      ext[0] > ext[1] ? (ext[0] > ext[2] ? 0 : 2) : ext[1] > ext[2] ? 1 : 2;

    // SAH binning on centroids
    const BINS = 20;
    const binCount = new Array(BINS).fill(0);
    const binMin = Array.from({ length: BINS }, () => [
      Infinity,
      Infinity,
      Infinity,
    ]);
    const binMax = Array.from({ length: BINS }, () => [
      -Infinity,
      -Infinity,
      -Infinity,
    ]);
    const invExt = ext[axis] > 0 ? 1.0 / ext[axis] : 0;

    for (let i = start; i < end; ++i) {
      const t = triOrder[i];
      let f = (centroids[3 * t + axis] - cMn[axis]) * invExt;
      let b = Math.floor(f * BINS);
      if (b < 0) b = 0;
      else if (b >= BINS) b = BINS - 1;
      binCount[b]++;
      for (let k = 0; k < 3; ++k) {
        const v = centroids[3 * t + k];
        binMin[b][k] = Math.min(binMin[b][k], v);
        binMax[b][k] = Math.max(binMax[b][k], v);
      }
    }

    // Prefix/suffix for SAH
    const leftCount = new Array(BINS).fill(0);
    const leftMin3 = Array.from({ length: BINS }, () => [
      Infinity,
      Infinity,
      Infinity,
    ]);
    const leftMax3 = Array.from({ length: BINS }, () => [
      -Infinity,
      -Infinity,
      -Infinity,
    ]);
    let accCnt = 0,
      accMin = [Infinity, Infinity, Infinity],
      accMax = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < BINS; ++i) {
      accCnt += binCount[i];
      leftCount[i] = accCnt;
      for (let k = 0; k < 3; ++k) {
        accMin[k] = Math.min(accMin[k], binMin[i][k]);
        accMax[k] = Math.max(accMax[k], binMax[i][k]);
        leftMin3[i][k] = accMin[k];
        leftMax3[i][k] = accMax[k];
      }
    }

    const rightCount = new Array(BINS).fill(0);
    const rightMin3 = Array.from({ length: BINS }, () => [
      Infinity,
      Infinity,
      Infinity,
    ]);
    const rightMax3 = Array.from({ length: BINS }, () => [
      -Infinity,
      -Infinity,
      -Infinity,
    ]);
    accCnt = 0;
    accMin = [Infinity, Infinity, Infinity];
    accMax = [-Infinity, -Infinity, -Infinity];
    for (let i = BINS - 1; i >= 0; --i) {
      accCnt += binCount[i];
      rightCount[i] = accCnt;
      for (let k = 0; k < 3; ++k) {
        accMin[k] = Math.min(accMin[k], binMin[i][k]);
        accMax[k] = Math.max(accMax[k], binMax[i][k]);
        rightMin3[i][k] = accMin[k];
        rightMax3[i][k] = accMax[k];
      }
    }

    // Parent surface area
    const PS =
      2 *
      ((mx[0] - mn[0]) * (mx[1] - mn[1]) +
        (mx[1] - mn[1]) * (mx[2] - mn[2]) +
        (mx[2] - mn[2]) * (mx[0] - mn[0]));

    // Find best split
    let bestCost = Infinity,
      bestBin = -1;
    for (let i = 0; i < BINS - 1; ++i) {
      const nL = leftCount[i],
        nR = rightCount[i + 1];
      if (nL === 0 || nR === 0) continue;
      const dL = [
        leftMax3[i][0] - leftMin3[i][0],
        leftMax3[i][1] - leftMin3[i][1],
        leftMax3[i][2] - leftMin3[i][2],
      ];
      const dR = [
        rightMax3[i + 1][0] - rightMin3[i + 1][0],
        rightMax3[i + 1][1] - rightMin3[i + 1][1],
        rightMax3[i + 1][2] - rightMin3[i + 1][2],
      ];
      const sL = 2 * (dL[0] * dL[1] + dL[1] * dL[2] + dL[2] * dL[0]);
      const sR = 2 * (dR[0] * dR[1] + dR[1] * dR[2] + dR[2] * dR[0]);
      const cost = (sL / PS) * nL + (sR / PS) * nR;
      if (cost < bestCost) {
        bestCost = cost;
        bestBin = i;
      }
    }

    // Fallback to median if no good split
    if (bestBin < 0) {
      const mid = start + (count >> 1);
      const l = recurse(start, mid);
      const r = recurse(mid, end);
      nodes[nodeIdx] = { mn, mx, left: l, right: r };
      return nodeIdx;
    }

    // Partition around split plane
    const splitPos = cMn[axis] + ((bestBin + 1) / BINS) * ext[axis];
    let i = start,
      j = end - 1;
    while (i <= j) {
      const t = triOrder[i];
      if (centroids[3 * t + axis] < splitPos) {
        i++;
      } else {
        [triOrder[i], triOrder[j]] = [triOrder[j], triOrder[i]];
        j--;
      }
    }
    const mid = i;

    // Recurse children
    const leftIdx = recurse(start, mid);
    const rightIdx = recurse(mid, end);
    nodes[nodeIdx] = { mn, mx, left: leftIdx, right: rightIdx };
    return nodeIdx;
  }

  recurse(0, tris);
  return { nodes, triOrder };
}

function padToVec4(array) {
  const byteLength = array.length * 4;
  const padBytes = (16 - (byteLength % 16)) % 16;
  const padFloats = padBytes / 4;
  if (padFloats === 0) return array;
  const padded = new Float32Array(array.length + padFloats);
  padded.set(array);
  return padded;
}

// Flatten BVH + triangles for SSBOs
function flattenForSSBO(mesh, bvh) {
  const tc = bvh.triOrder.length;
  const triPos = new Float32Array(tc * 3 * 4),
    triNor = new Float32Array(tc * 3 * 4);
  for (let i = 0; i < tc; i++) {
    const ti = bvh.triOrder[i];
    for (let v = 0; v < 3; v++) {
      const vi = mesh.vertIdx[3 * ti + v],
        ni = mesh.normIdx[3 * ti + v];
      const dst = (i * 3 + v) * 4;
      triPos[dst + 0] = mesh.positions[3 * vi + 0];
      triPos[dst + 1] = mesh.positions[3 * vi + 1];
      triPos[dst + 2] = mesh.positions[3 * vi + 2];
      triPos[dst + 3] = 0.0; // padding
      triNor[dst + 0] = mesh.normals[3 * ni + 0];
      triNor[dst + 1] = mesh.normals[3 * ni + 1];
      triNor[dst + 2] = mesh.normals[3 * ni + 2];
      triNor[dst + 3] = 0.0; // padding
    }
  }
  const nc = bvh.nodes.length;
  const bvhData = new Float32Array(nc * 8);
  for (let i = 0; i < nc; i++) {
    const n = bvh.nodes[i],
      b = 8 * i;
    bvhData[b + 0] = n.mn[0];
    bvhData[b + 1] = n.mn[1];
    bvhData[b + 2] = n.mn[2];
    bvhData[b + 3] = n.left;
    bvhData[b + 4] = n.mx[0];
    bvhData[b + 5] = n.mx[1];
    bvhData[b + 6] = n.mx[2];
    bvhData[b + 7] = n.right;
  }
  return { triPos, triNor, bvhData };
}

// Split a Float32Array in half
function splitHalf(a) {
  const count = a.length / 4,
    half = Math.ceil(count / 2);
  return [a.slice(0, half * 4), a.slice(half * 4)];
}

// Create a GPU STORAGE buffer from a TypedArray
function makeBuf(arr) {
  const buf = device.createBuffer({
    size: Math.max(arr.byteLength, 16), // Minimum size
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buf, 0, arr.buffer, arr.byteOffset, arr.byteLength);
  return buf;
}

// Mouse‐look handler
function onMouseMove(e) {
  const s = 0.002;
  yaw += e.movementX * s;
  pitch = Math.max(
    -Math.PI / 2 + 0.01,
    Math.min(Math.PI / 2 - 0.01, pitch + e.movementY * s)
  );
}

// Update camera pos & return look‐at vectors
function updateCamera() {
  const cp = Math.cos(pitch),
    sp = Math.sin(pitch);
  const cy = Math.cos(yaw),
    sy = Math.sin(yaw);
  const fwd = [cp * sy, sp, -cp * cy];
  const rightVec = [cy, 0, sy];
  const speed = 0.05;
  if (keys.includes("w"))
    cameraPos = cameraPos.map((v, i) => v + fwd[i] * speed);
  if (keys.includes("s"))
    cameraPos = cameraPos.map((v, i) => v - fwd[i] * speed);
  if (keys.includes("a"))
    cameraPos = cameraPos.map((v, i) => v - rightVec[i] * speed);
  if (keys.includes("d"))
    cameraPos = cameraPos.map((v, i) => v + rightVec[i] * speed);
  return {
    pos: cameraPos,
    tgt: [cameraPos[0] + fwd[0], cameraPos[1] + fwd[1], cameraPos[2] + fwd[2]],
    up: [0, 1, 0],
  };
}

// Inline "blit" WGSL to sample outputTex
const blitWGSL = `
@group(0) @binding(0) var myTex: texture_2d<f32>;
@group(0) @binding(1) var mySamp: sampler;

struct VSOut {
  @builtin(position) Position: vec4<f32>,
  @location(0)       uv:       vec2<f32>,
}

@vertex
fn vs_main(@location(0) pos: vec2<f32>) -> VSOut {
  var o: VSOut;
  o.Position = vec4<f32>(pos,0.0,1.0);
  o.uv = pos * vec2<f32>(0.5,-0.5) + vec2<f32>(0.5,0.5);
  return o;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
  return textureSample(myTex, mySamp, in.uv);
}
`;

// Generate a simple test scene
function generateTestScene() {
  // Create a simple quad
  const positions = new Float32Array([
    -1,
    -1,
    0, // v0
    1,
    -1,
    0, // v1
    1,
    1,
    0, // v2
    -1,
    1,
    0, // v3
  ]);

  const normals = new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]);

  const vertIdx = new Uint32Array([
    0,
    1,
    2, // first triangle
    0,
    2,
    3, // second triangle
  ]);

  const normIdx = new Uint32Array([0, 1, 2, 0, 2, 3]);

  return {
    positions,
    normals,
    vertIdx,
    normIdx,
  };
}

// Math helpers
function normalize(v) {
  const l = Math.hypot(...v) || 1;
  return v.map((x) => x / l);
}

function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

window.addEventListener("DOMContentLoaded", () => {
  (async () => {
    try {
      // Check WebGPU support
      if (!navigator.gpu) {
        document.getElementById("fpsCounter").textContent =
          "WebGPU not supported";
        return;
      }

      // 1) Canvas & WebGPU setup
      const canvas = document.getElementById("gpuCanvas");
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;

      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        document.getElementById("fpsCounter").textContent = "No WebGPU adapter";
        return;
      }

      device = await adapter.requestDevice();
      context = canvas.getContext("webgpu");
      const swapFormat = navigator.gpu.getPreferredCanvasFormat();
      context.configure({ device, format: swapFormat, alphaMode: "opaque" });

      // 2) Load & normalize mesh (fallback to test scene)
      let raw;
      try {
        raw = await loadObj("dragon.obj").then(normalizeMesh);
      } catch (e) {
        console.log("Using generated test scene");
        raw = generateTestScene();
      }

      // 3) Build LBVH & flatten
      document.getElementById("fpsCounter").textContent = "Building BVH…";
      await initMortonPipeline();
      const bvh = buildLBVH(raw, 1);
      const { triPos, triNor, bvhData } = flattenForSSBO(raw, bvh);
      const [p0, p1] = splitHalf(triPos);
      const [n0, n1] = splitHalf(triNor);
      const [b0, b1] = splitHalf(bvhData);

      console.log("Mesh stats:", {
        triangles: raw.vertIdx.length / 3,
        nodes: bvh.nodes.length,
        p0_count: p0.length / 4,
        p1_count: p1.length / 4,
      });

      // 4) Upload buffers
      pos0Buf = makeBuf(padToVec4(p0));
      pos1Buf = makeBuf(padToVec4(p1));
      nor0Buf = makeBuf(padToVec4(n0));
      nor1Buf = makeBuf(padToVec4(n1));
      bvh0Buf = makeBuf(padToVec4(b0));
      bvh1Buf = makeBuf(padToVec4(b1));

      // 5) Uniform buffer
      uniformBuf = device.createBuffer({
        size: 4 * (4 + 4 + 4 + 4 + 4) + 16 * 4 + 4 * 4 + 4 * 4,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });

      // 6) Output texture
      sampler = device.createSampler({
        magFilter: "linear",
        minFilter: "linear",
      });
      outputTex = device.createTexture({
        size: {
          width: canvas.width,
          height: canvas.height,
          depthOrArrayLayers: 1,
        },
        format: "rgba8unorm",
        usage:
          GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      });

      // 7) Quad vertex buffer
      quadBuf = device.createBuffer({
        size: 2 * 4 * 6,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(
        quadBuf,
        0,
        new Float32Array([-1, -1, 1, -1, 1, 1, -1, -1, 1, 1, -1, 1])
      );

      // 8) Compute pipeline
      let computeCode = await loadText("compute.wgsl").then((s) =>
        s
          .replaceAll("__COUNT_POS0__", `${p0.length / 4}u`)
          .replaceAll("__COUNT_NOR0__", `${n0.length / 4}u`)
          .replaceAll("__COUNT_BVH0__", `${b0.length / 4}u`)
      );

      console.log("Loading compute shader...");
      const computeModule = device.createShaderModule({ code: computeCode });
      computePipeline = device.createComputePipeline({
        layout: "auto",
        compute: { module: computeModule, entryPoint: "cs_main" },
      });

      computeBindGroup = device.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: pos0Buf } },
          { binding: 1, resource: { buffer: pos1Buf } },
          { binding: 2, resource: { buffer: nor0Buf } },
          { binding: 3, resource: { buffer: nor1Buf } },
          { binding: 4, resource: { buffer: bvh0Buf } },
          { binding: 5, resource: { buffer: bvh1Buf } },
          { binding: 6, resource: { buffer: uniformBuf } },
          { binding: 7, resource: outputTex.createView() },
        ],
      });

      // 9) Blit pipeline
      const blitModule = device.createShaderModule({ code: blitWGSL });
      blitPipeline = device.createRenderPipeline({
        layout: "auto",
        vertex: {
          module: blitModule,
          entryPoint: "vs_main",
          buffers: [
            {
              arrayStride: 8,
              attributes: [
                { shaderLocation: 0, offset: 0, format: "float32x2" },
              ],
            },
          ],
        },
        fragment: {
          module: blitModule,
          entryPoint: "fs_main",
          targets: [{ format: swapFormat }],
        },
        primitive: { topology: "triangle-list" },
      });

      blitBindGroup = device.createBindGroup({
        layout: blitPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: outputTex.createView() },
          { binding: 1, resource: sampler },
        ],
      });

      // 10) Input
      window.addEventListener("keydown", (e) => {
        if (!keys.includes(e.key.toLowerCase())) keys.push(e.key.toLowerCase());
      });
      window.addEventListener("keyup", (e) => {
        keys = keys.filter((k) => k !== e.key.toLowerCase());
      });
      canvas.onclick = () => canvas.requestPointerLock();
      document.addEventListener("pointerlockchange", () => {
        if (document.pointerLockElement === canvas) {
          document.addEventListener("mousemove", onMouseMove);
        } else {
          document.removeEventListener("mousemove", onMouseMove);
        }
      });

      // 11) Render loop
      let lastTime = performance.now(),
        frameCount = 0;
      let lastFrameCount = 0;

      function frame(nowMS) {
        try {
          // FPS
          frameCount++;
          if (performance.now() - lastTime >= 1000) {
            const fps = (
              (frameCount - lastFrameCount) /
              ((performance.now() - lastTime) / 1000)
            ).toFixed(1);
            document.getElementById("fpsCounter").textContent = `FPS: ${fps}`;
            lastTime = performance.now();
            lastFrameCount = frameCount;
          }

          // Update camera uniforms
          const { pos, tgt, up } = updateCamera();
          const forward = normalize([
            tgt[0] - pos[0],
            tgt[1] - pos[1],
            tgt[2] - pos[2],
          ]);
          const right = normalize(cross(forward, up));
          const trueUp = cross(right, forward);
          const fov = Math.PI / 3,
            aspect = canvas.width / canvas.height,
            fl = 1 / Math.tan(fov / 2);
          const camRightVec = right.map((v) => v * fl * aspect);
          const camUpVec = trueUp.map((v) => v * fl);
          const camForwardVec = forward.map((v) => v * fl);
          let modelMat = updateTRSMatrix();
          let modelMatInv = new Float32Array(16);
          invert4(modelMatInv, modelMat);

          // Pack uniforms
          const udata = new Float32Array([
            pos[0],
            pos[1],
            pos[2],
            0,
            5,
            -5,
            5,
            0.3,
            camRightVec[0],
            camRightVec[1],
            camRightVec[2],
            0,
            camUpVec[0],
            camUpVec[1],
            camUpVec[2],
            0,
            camForwardVec[0],
            camForwardVec[1],
            camForwardVec[2],
            0,
            ...modelMatInv,
            canvas.width,
            canvas.height,
            0,
            0,
            frameCount,
            0,
            0,
            0,
          ]);
          device.queue.writeBuffer(uniformBuf, 0, udata);

          // Encode
          const enc = device.createCommandEncoder();

          // Compute pass
          {
            const pass = enc.beginComputePass();
            pass.setPipeline(computePipeline);
            pass.setBindGroup(0, computeBindGroup);
            pass.dispatchWorkgroups(
              Math.ceil(canvas.width / 16),
              Math.ceil(canvas.height / 16)
            );
            pass.end();
          }

          // Blit pass
          {
            const rp = enc.beginRenderPass({
              colorAttachments: [
                {
                  view: context.getCurrentTexture().createView(),
                  loadOp: "clear",
                  clearValue: { r: 0, g: 0, b: 0, a: 1 },
                  storeOp: "store",
                },
              ],
            });
            rp.setPipeline(blitPipeline);
            rp.setBindGroup(0, blitBindGroup);
            rp.setVertexBuffer(0, quadBuf);
            rp.draw(6);
            rp.end();
          }

          device.queue.submit([enc.finish()]);
          requestAnimationFrame(frame);
        } catch (err) {
          console.error("Frame error:", err);
          document.getElementById(
            "fpsCounter"
          ).textContent = `Error: ${err.message}`;
        }
      }

      console.log("Starting render loop...");
      requestAnimationFrame(frame);
    } catch (err) {
      console.error("Initialization error:", err);
      document.getElementById(
        "fpsCounter"
      ).textContent = `Init Error: ${err.message}`;
    }
  })();
});
