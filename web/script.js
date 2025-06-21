// script.js

// — Global state —
let device, context, pipeline, bindGroup, quadBuf, uniformBuf;
let cameraPos = [0, 0, 5], yaw = 0, pitch = 0, keys = [];
let scale = [1, 1, 1], rotation = [0, 0, 0], pos = [0, 0, 0];

// — Helper: load text over HTTP —
async function loadText(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Failed to load ${url}`);
  return r.text();
}

// — Helper: load a Wavefront OBJ (positions, normals, face-indices) —
async function loadObj(url) {
  const txt = await loadText(url);
  const pos = [], nor = [], vIdx = [], nIdx = [];
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
      const refs = line.split(/\s+/).slice(1).map(p => p.split("/"));
      const vC = pos.length / 3, nC = nor.length / 3;
      for (let i = 1; i < refs.length - 1; i++) {
        const [v0,,n0] = refs[0], [v1,,n1] = refs[i], [v2,,n2] = refs[i+1];
        vIdx.push(toIndex(v0, vC), toIndex(v1, vC), toIndex(v2, vC));
        nIdx.push(toIndex(n0, nC), toIndex(n1, nC), toIndex(n2, nC));
      }
    }
  }
  return {
    positions: new Float32Array(pos),
    normals:   new Float32Array(nor),
    vertIdx:   new Uint32Array(vIdx),
    normIdx:   new Uint32Array(nIdx),
  };
}

// — Helper: center & scale mesh into [-1,1]^3 —
function normalizeMesh(m) {
  const p = m.positions, n = p.length/3;
  const mn = [Infinity,Infinity,Infinity], mx = [-Infinity,-Infinity,-Infinity];
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 3; k++) {
      const v = p[3*i + k];
      mn[k] = Math.min(mn[k], v);
      mx[k] = Math.max(mx[k], v);
    }
  }
  const ctr = mn.map((v,i) => (v + mx[i]) * 0.5);
  const sc  = 1 / Math.max(mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2]);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 3; k++) {
      p[3*i + k] = (p[3*i + k] - ctr[k]) * sc;
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
  console.log("--- loaded Morton shader ---\n", code, "\n----------------------------");
  const module = device.createShaderModule({ code });
  mortonPipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "main" },
    label: "MortonPipeline"
  });
  mortonBindGroupLayout = mortonPipeline.getBindGroupLayout(0);
}

// — Helper: upload TypedArray as STORAGE buffer —
function createStorageBufferFromArray(arr) {
  const buf = device.createBuffer({
    size: arr.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  });
  device.queue.writeBuffer(buf, 0, arr.buffer, arr.byteOffset, arr.byteLength);
  return buf;
}

// — Helper: read back GPU buffer to ArrayBuffer —
async function readbackBuffer(srcBuf, size) {
  const dst = device.createBuffer({
    size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
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
 * Build LBVH:
 *  - GPU does centroid→normalize→quantize→Morton
 *  - CPU only: sort codes & recurse
 */
async function buildLBVH(mesh, leaf = 1) {
  const tris = mesh.vertIdx.length / 3;
  // --- CPU: compute bounds over all vertex positions ---
  const posArr = mesh.positions;
  const mn = [ Infinity, Infinity, Infinity ];
  const mx = [ -Infinity, -Infinity, -Infinity ];
  for (let i = 0; i < posArr.length; i += 3) {
    mn[0] = Math.min(mn[0], posArr[i]);
    mn[1] = Math.min(mn[1], posArr[i+1]);
    mn[2] = Math.min(mn[2], posArr[i+2]);
    mx[0] = Math.max(mx[0], posArr[i]);
    mx[1] = Math.max(mx[1], posArr[i+1]);
    mx[2] = Math.max(mx[2], posArr[i+2]);
  }
  const extent = [ mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2] ];
  const invExt = extent.map(e => 1.0 / (e || 1e-6));

  // --- Upload buffers / uniform ---
  // number of vertices in the mesh
  const vertexCount = mesh.positions.length / 3;

  // make a Float32Array that’s 4 floats per vertex
  const padded = new Float32Array(vertexCount * 4);
  for (let i = 0; i < vertexCount; i++) {
    padded[4*i  ] = mesh.positions[3*i];
    padded[4*i+1] = mesh.positions[3*i+1];
    padded[4*i+2] = mesh.positions[3*i+2];
    padded[4*i+3] = 0;       // the “w” component
  }

  // now upload that
  const posBuf = createStorageBufferFromArray(padded);
  const idxBuf = createStorageBufferFromArray(mesh.vertIdx);
  const bdat = new Float32Array([
    mn[0], mn[1], mn[2], 0,
    invExt[0], invExt[1], invExt[2], 0
  ]);
  const boundsBuf = device.createBuffer({
    size: bdat.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  });
  device.queue.writeBuffer(boundsBuf, 0, bdat.buffer, bdat.byteOffset, bdat.byteLength);

  const mortonBuf = device.createBuffer({
    size: tris * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
  });

  // --- Dispatch compute ---
  const bg = device.createBindGroup({
    layout: mortonBindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: posBuf     } },
      { binding: 1, resource: { buffer: idxBuf     } },
      { binding: 2, resource: { buffer: boundsBuf  } },
      { binding: 3, resource: { buffer: mortonBuf  } }
    ]
  });
  {
    const enc  = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(mortonPipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(tris / 64));
    pass.end();
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
  }

  // --- Read back codes & clean up ---
  const raw       = await readbackBuffer(mortonBuf, tris * 4);
  const mortonArr = new Uint32Array(raw);
  posBuf.destroy();
  idxBuf.destroy();
  boundsBuf.destroy();
  mortonBuf.destroy();

  console.log("GPU Morton codes (first 8):", mortonArr.slice(0,8));

  // --- CPU: sort & build BVH nodes ---
  const ord = new Uint32Array(tris);
  for (let i = 0; i < tris; i++) ord[i] = i;
  ord.sort((a,b) => mortonArr[a] - mortonArr[b]);
  const triOrder = Array.from(ord);

  const nodes = [];
  function recurse(s, e) {
    const idx = nodes.length; nodes.push(null);
    const count = e - s;
    const mn = [Infinity,Infinity,Infinity], mx = [-Infinity,-Infinity,-Infinity];
    for (let i = s; i < e; i++) {
      const ti = triOrder[i];
      for (let v = 0; v < 3; v++) {
        const off = mesh.vertIdx[3*ti + v] * 3;
        for (let k = 0; k < 3; k++) {
          const val = mesh.positions[off + k];
          mn[k] = Math.min(mn[k], val);
          mx[k] = Math.max(mx[k], val);
        }
      }
    }
    if (count <= leaf) {
      nodes[idx] = { mn, mx, left: s, right: -count };
    } else {
      const mid      = (s + e) >> 1;
      const leftIdx  = recurse(s, mid);
      const rightIdx = recurse(mid, e);
      nodes[idx]   = { mn, mx, left: leftIdx, right: rightIdx };
    }
    return idx;
  }
  recurse(0, tris);
  return { nodes, triOrder };
}

// — Flatten BVH + triangles for SSBOs used by fragment shader —
function flattenForSSBO(mesh, bvh) {
  const tc = bvh.triOrder.length;
  const triPos = new Float32Array(tc*3*4), triNor = new Float32Array(tc*3*4);
  for (let i = 0; i < tc; i++) {
    const ti = bvh.triOrder[i];
    for (let v = 0; v < 3; v++) {
      const vi = mesh.vertIdx[3*ti + v], ni = mesh.normIdx[3*ti + v];
      const dst = (i*3 + v) * 4;
      triPos[dst]   = mesh.positions[3*vi];
      triPos[dst+1] = mesh.positions[3*vi+1];
      triPos[dst+2] = mesh.positions[3*vi+2];
      triNor[dst]   = mesh.normals  [3*ni];
      triNor[dst+1] = mesh.normals  [3*ni+1];
      triNor[dst+2] = mesh.normals  [3*ni+2];
    }
  }
  const nc = bvh.nodes.length;
  const bvhData = new Float32Array(nc*8);
  for (let i = 0; i < nc; i++) {
    const n = bvh.nodes[i], b = 8*i;
    bvhData[b]   = n.mn[0];
    bvhData[b+1] = n.mn[1];
    bvhData[b+2] = n.mn[2];
    bvhData[b+3] = n.left;
    bvhData[b+4] = n.mx[0];
    bvhData[b+5] = n.mx[1];
    bvhData[b+6] = n.mx[2];
    bvhData[b+7] = n.right;
  }
  return { triPos, triNor, bvhData };
}

function splitHalf(a) {
  const vc = a.length / 4, h = Math.ceil(vc/2);
  return [ a.slice(0, h*4), a.slice(h*4) ];
}
function makeBuf(arr) {
  const buf = device.createBuffer({
    size: arr.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  });
  device.queue.writeBuffer(buf, 0, arr.buffer, arr.byteOffset, arr.byteLength);
  return buf;
}

// — Camera controls & math helpers —
function onMouseMove(e) {
  const s = 0.002;
  yaw   += e.movementX * s;
  pitch = Math.max(-Math.PI/2 + 0.01, Math.min(Math.PI/2 - 0.01, pitch - e.movementY * s));
}
function updateCamera() {
  const cp = Math.cos(pitch), sp = Math.sin(pitch),
        cy = Math.cos(yaw),   sy = Math.sin(yaw);
  const fwd      = [ cp*sy, sp, -cp*cy ];
  const rightVec = [ cy, 0, sy ];
  const speed    = 0.05;
  if (keys.includes("w")) cameraPos = cameraPos.map((v,i) => v + fwd[i]*speed);
  if (keys.includes("s")) cameraPos = cameraPos.map((v,i) => v - fwd[i]*speed);
  if (keys.includes("a")) cameraPos = cameraPos.map((v,i) => v - rightVec[i]*speed);
  if (keys.includes("d")) cameraPos = cameraPos.map((v,i) => v + rightVec[i]*speed);
  return {
    pos: cameraPos,
    tgt: [ cameraPos[0]+fwd[0], cameraPos[1]+fwd[1], cameraPos[2]+fwd[2] ],
    up:  [0,1,0]
  };
}
function invertMat4(m) {
  const out = new Float32Array(16), a = m;
  const [a00,a01,a02,a03,
         a10,a11,a12,a13,
         a20,a21,a22,a23,
         a30,a31,a32,a33] = a;
  const b00 = a00*a11 - a01*a10;
  const b01 = a00*a12 - a02*a10;
  const b02 = a00*a13 - a03*a10;
  const b03 = a01*a12 - a02*a11;
  const b04 = a01*a13 - a03*a11;
  const b05 = a02*a13 - a03*a12;
  const b06 = a20*a31 - a21*a30;
  const b07 = a20*a32 - a22*a30;
  const b08 = a20*a33 - a23*a30;
  const b09 = a21*a32 - a22*a31;
  const b10 = a21*a33 - a23*a31;
  const b11 = a22*a33 - a23*a32;
  const det = b00*b11 - b01*b10 + b02*b09 + b03*b08 - b04*b07 + b05*b06;
  if (!det) return null;
  const idet = 1.0 / det;
  out[0 ] = (a11*b11 - a12*b10 + a13*b09)*idet;
  out[1 ] = (-a01*b11 + a02*b10 - a03*b09)*idet;
  out[2 ] = (a31*b05 - a32*b04 + a33*b03)*idet;
  out[3 ] = 0;
  out[4 ] = (-a10*b11 + a12*b08 - a13*b07)*idet;
  out[5 ] = (a00*b11 - a02*b08 + a03*b07)*idet;
  out[6 ] = (-a30*b05 + a32*b02 - a33*b01)*idet;
  out[7 ] = 0;
  out[8 ] = (a10*b10 - a11*b08 + a13*b06)*idet;
  out[9 ] = (-a00*b10 + a01*b08 - a03*b06)*idet;
  out[10] = (a30*b04 - a31*b02 + a33*b00)*idet;
  out[11] = 0;
  out[12] = (-a10*b09 + a11*b07 - a12*b06)*idet;
  out[13] = (a00*b09 - a01*b07 + a02*b06)*idet;
  out[14] = (-a30*b03 + a31*b01 - a32*b00)*idet;
  out[15] = 1;
  return out;
}

// — Render loop —
let lastTime = performance.now(), frameCount = 0;
function frame(nowMS) {
  const now = nowMS * 0.001;
  frameCount++;
  if (performance.now() - lastTime >= 1000) {
    const fps = (frameCount / ((performance.now()-lastTime)/1000)).toFixed(1);
    document.getElementById("fpsCounter").textContent = `FPS: ${fps}`;
    lastTime = performance.now();
    frameCount = 0;
  }

  const { pos: camPos, tgt, up } = updateCamera();
  const normalizeVec = v => { const l = Math.hypot(...v)||1; return v.map(x=>x/l) };
  const cross = (a,b)=>[ a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0] ];
  const fwd   = normalizeVec([ tgt[0]-camPos[0], tgt[1]-camPos[1], tgt[2]-camPos[2] ]);
  const rightVec = normalizeVec(cross(fwd, up));
  const trueUp  = cross(rightVec, fwd);

  const fov = Math.PI/4, aspect = innerWidth/innerHeight, fl = 1/Math.tan(fov*0.5);
  const camRight   = rightVec.map(v=>v*fl*aspect);
  const camUpVec   = trueUp.map(v=>v*fl);
  const camForwVec = fwd.map(v=>v*fl);

  // Build model matrix
  const [sx,sy,sz]=scale, [rx,ry,rz]=rotation, [tx,ty,tz]=pos;
  const cx=Math.cos(rx), sx_=Math.sin(rx);
  const cy=Math.cos(ry), sy_=Math.sin(ry);
  const cz=Math.cos(rz), sz_=Math.sin(rz);
  const r00=cy*cz, r01=-cy*sz_, r02=sy_;
  const r10=sx_*sy_*cz+cx*sz_, r11=-sx_*sy_*sz_+cx*cz, r12=-sx_*cy;
  const r20=-cx*sy_*cz+sx_*sz_, r21=cx*sy_*sz_+sx_*cz, r22=cx*cy;
  const m00=r00*sx, m01=r01*sx, m02=r02*sx;
  const m10=r10*sy, m11=r11*sy, m12=r12*sy;
  const m20=r20*sz, m21=r21*sz, m22=r22*sz;
  const modelMat=[
    m00,m10,m20,0,
    m01,m11,m21,0,
    m02,m12,m22,0,
    tx, ty, tz, 1
  ];

  const udat = new Float32Array([
    camPos[0], camPos[1], camPos[2], 1,
    5, -5, -5, 0.5,
    camRight[0], camRight[1], camRight[2], 0,
    camUpVec[0], camUpVec[1], camUpVec[2], 0,
    camForwVec[0], camForwVec[1], camForwVec[2], 0,
    ...invertMat4(modelMat)
  ]);
  device.queue.writeBuffer(uniformBuf, 0, udat);

  const cmd  = device.createCommandEncoder();
  const pass = cmd.beginRenderPass({
    colorAttachments:[{
      view: context.getCurrentTexture().createView(),
      loadOp:"clear",
      clearValue:{r:0.6,g:0.8,b:1,a:1},
      storeOp:"store"
    }]
  });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.setVertexBuffer(0, quadBuf);
  pass.draw(6);
  pass.end();
  device.queue.submit([cmd.finish()]);

  requestAnimationFrame(frame);
}

// — Initialization: DOM ready →
window.addEventListener("DOMContentLoaded", () => {
  (async () => {
    // Setup WebGPU
    const canvas = document.getElementById("gpuCanvas");
    canvas.width  = innerWidth;
    canvas.height = innerHeight;
    if (!navigator.gpu) throw new Error("WebGPU not supported");
    const adapter = await navigator.gpu.requestAdapter();
    device  = await adapter.requestDevice();
    context = canvas.getContext("webgpu");
    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({ device, format, alphaMode: "opaque" });

    // Morton compute pipeline
    await initMortonPipeline();

    // Load + normalize mesh
    const raw = await loadObj("dragon.obj").then(normalizeMesh);

    // Build LBVH
    const start = performance.now();
    const bvh = await buildLBVH(raw, 1);
    const end = performance.now();
    console.log((end - start) / 1000);

    // Flatten BVH + triangle data for fragment shader
    const { triPos, triNor, bvhData } = flattenForSSBO(raw, bvh);
    const [p0,p1] = splitHalf(triPos);
    const [n0,n1] = splitHalf(triNor);
    const [b0,b1] = splitHalf(bvhData);
    const pos0Buf = makeBuf(p0);
    const pos1Buf = makeBuf(p1);
    const nor0Buf = makeBuf(n0);
    const nor1Buf = makeBuf(n1);
    const bvh0Buf = makeBuf(b0);
    const bvh1Buf = makeBuf(b1);

    // Uniform buffer for rendering
    uniformBuf = device.createBuffer({
      size: 4*4*9,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
    });

    // Build render pipeline
    const wgsl = await loadText("shader.wgsl");
    const code = wgsl
      .replaceAll("__COUNT_POS0__", `${p0.length/4}u`)
      .replaceAll("__COUNT_NOR0__", `${n0.length/4}u`)
      .replaceAll("__COUNT_BVH0__", `${b0.length/4}u`);
    const module = device.createShaderModule({ code });
    const quad = new Float32Array([-1,-1, 1,-1, 1,1, -1,-1, 1,1, -1,1]);
    quadBuf = device.createBuffer({
      size: quad.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
    });
    device.queue.writeBuffer(quadBuf, 0, quad);

    const bgl = device.createBindGroupLayout({
      entries:[
        {binding:0,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}},
        {binding:1,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}},
        {binding:2,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}},
        {binding:3,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}},
        {binding:4,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}},
        {binding:5,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"read-only-storage"}},
        {binding:6,visibility:GPUShaderStage.FRAGMENT,buffer:{type:"uniform"}},
      ]
    });
    pipeline = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts:[bgl] }),
      vertex: { module, entryPoint:"vs_main", buffers:[
        { arrayStride:8, attributes:[{shaderLocation:0,offset:0,format:"float32x2"}] }
      ]},
      fragment:{ module, entryPoint:"fs_main", targets:[{format}] },
      primitive:{ topology:"triangle-list" }
    });
    bindGroup = device.createBindGroup({
      layout:bgl,
      entries:[
        {binding:0, resource:{buffer:pos0Buf}},
        {binding:1, resource:{buffer:pos1Buf}},
        {binding:2, resource:{buffer:nor0Buf}},
        {binding:3, resource:{buffer:nor1Buf}},
        {binding:4, resource:{buffer:bvh0Buf}},
        {binding:5, resource:{buffer:bvh1Buf}},
        {binding:6, resource:{buffer:uniformBuf}},
      ]
    });

    // Camera & input
    window.addEventListener("keydown", e=>{ if(!keys.includes(e.key)) keys.push(e.key); });
    window.addEventListener("keyup",   e=>{ keys=keys.filter(k=>k!==e.key); });
    canvas.onclick = () => canvas.requestPointerLock();
    document.addEventListener("pointerlockchange", ()=>{
      if (document.pointerLockElement === canvas) {
        document.addEventListener("mousemove", onMouseMove);
      } else {
        document.removeEventListener("mousemove", onMouseMove);
      }
    });
    ["scaleX","scaleY","scaleZ"].forEach((id,i)=>{
      document.getElementById(id).addEventListener("input", e=>{
        scale[i] = Math.max(0.01, parseFloat(e.target.value));
      });
    });
    ["rotX","rotY","rotZ"].forEach((id,i)=>{
      document.getElementById(id).addEventListener("input", e=>{
        rotation[i] = parseFloat(e.target.value) * Math.PI/180;
      });
    });
    ["posX","posY","posZ"].forEach((id,i)=>{
      document.getElementById(id).addEventListener("input", e=>{
        pos[i] = parseFloat(e.target.value);
      });
    });

    // Start loop
    requestAnimationFrame(frame);
  })();
});