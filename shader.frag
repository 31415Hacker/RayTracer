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
  int idx = i*3+v;
  ivec2 p = ivec2(idx%u_texSize, idx/u_texSize);
  return texelFetch(u_triTex,p,0).xyz;
}
// fetch vertex normal
vec3 fetchNor(int i,int v){
  int idx = i*3+v;
  ivec2 p = ivec2(idx%u_texSize, idx/u_texSize);
  return normalize(texelFetch(u_normTex,p,0).xyz);
}

// fetch BVH node (2 pixels)
void fetchNode(int ni, out vec3 mn, out vec3 mx, out int left, out int rc){
  int pix0 = ni*2;
  ivec2 p0 = ivec2(pix0%u_nodeTexSize, pix0/u_nodeTexSize);
  vec4 d0  = texelFetch(u_nodeTex,p0,0);
  int pix1 = pix0+1;
  ivec2 p1 = ivec2(pix1%u_nodeTexSize, pix1/u_nodeTexSize);
  vec4 d1  = texelFetch(u_nodeTex,p1,0);

  mn    = d0.rgb;
  left  = int(d0.a + 0.5);
  mx    = d1.rgb;
  rc    = int(d1.a + (d1.a<0.0?-0.5:0.5));
}

// AABB intersect → returns true + tmin,tmax
bool intersectAABB(vec3 ro,vec3 invRd,vec3 mn,vec3 mx,
                   out float tmin, out float tmax){
  vec3 t0 = (mn - ro)*invRd;
  vec3 t1 = (mx - ro)*invRd;
  vec3 tmin3 = min(t0,t1);
  vec3 tmax3 = max(t0,t1);
  tmin = max(max(tmin3.x,tmin3.y),tmin3.z);
  tmax = min(min(tmax3.x,tmax3.y),tmax3.z);
  return tmax >= max(tmin,0.0);
}

// Möller–Trumbore with barycentrics
bool triIntersect(vec3 ro,vec3 rd,vec3 v0,vec3 v1,vec3 v2,
                  out float t,out float u,out float v){
  const float EPS=1e-6;
  vec3 e1=v1-v0, e2=v2-v0;
  vec3 p=cross(rd,e2);
  float a=dot(e1,p);
  if(abs(a)<EPS) return false;
  float invA=1.0/a;
  vec3 s=ro-v0;
  u=invA*dot(s,p);
  if(u<0.0||u>1.0) return false;
  vec3 q=cross(s,e1);
  v=invA*dot(rd,q);
  if(v<0.0||u+v>1.0) return false;
  float tt=invA*dot(e2,q);
  if(tt>EPS){ t=tt; return true; }
  return false;
}

vec3 raytrace(vec2 uv){
  // build camera ray in world
  vec3 fwd=normalize(u_cameraTarget-u_cameraPos);
  vec3 rt =normalize(cross(fwd,u_cameraUp));
  vec3 up =cross(rt,fwd);
  uv = uv*2.0 - 1.0;
  uv.x *= u_aspect;
  float fl = 1.0/tan(u_fov*0.5);
  vec3 ro_w = u_cameraPos;
  vec3 rd_w = normalize(fwd*fl + rt*uv.x + up*uv.y);

  // to object space
  mat3 invR = transpose(mat3(u_model));
  vec3 ro = invR * ro_w;
  vec3 rd = normalize(invR * rd_w);
  vec3 light = invR * vec3(5.0,5.0,5.0);
  vec3 invRd = 1.0/rd;

  // BVH traversal stack
  const int MAX_ST=64;
  int stack[MAX_ST];
  int sp=0; stack[sp++]=0; // start at root

  float bestT = 1e20;
  vec3  bestN = vec3(0.0);

  while(sp>0){
    int ni = stack[--sp];
    vec3 mn,mx; int left,rc;
    fetchNode(ni,mn,mx,left,rc);

    float tmin,tmax;
    if(!intersectAABB(ro,invRd,mn,mx,tmin,tmax)) continue;

    if (tmin > bestT) continue;

    if (rc < 0) {
        // leaf: left=triStart, rc=-count
        int start = left;
        int cnt   = -rc;
        for (int j = 0; j < cnt; ++j) {
            int ti = start + j;

            // 1) PREFETCH the 3 vertex positions
            vec3 v0 = fetchPos(ti, 0);
            vec3 v1 = fetchPos(ti, 1);
            vec3 v2 = fetchPos(ti, 2);

            // 2) PREFETCH the 3 vertex normals
            vec3 n0 = fetchNor(ti, 0);
            vec3 n1 = fetchNor(ti, 1);
            vec3 n2 = fetchNor(ti, 2);

            // 3) do intersection
            float t, u, v;
            if (triIntersect(ro, rd, v0, v1, v2, t, u, v) && t < bestT) {
                bestT = t;
                // 4) smooth‐shade with the already‐fetched normals
                bestN = normalize(n0 * (1.0 - u - v)
                                + n1 * u
                                + n2 * v);
            }
        }
    } else {
      // internal: left & rc are children
      int c0=left, c1=rc;
      // fetch their AABBs + entry ts
      vec3 mn0,mx0,mn1,mx1; int dummy;
      float t0min,t0max,t1min,t1max;
      fetchNode(c0,mn0,mx0,dummy,dummy);
      fetchNode(c1,mn1,mx1,dummy,dummy);
      bool hit0 = intersectAABB(ro,invRd,mn0,mx0,t0min,t0max);
      bool hit1 = intersectAABB(ro,invRd,mn1,mx1,t1min,t1max);

      if(hit0 && hit1){
        // push closest first
        if(t0min > t1min){
          stack[sp++] = c1;
          stack[sp++] = c0;
        } else {
          stack[sp++] = c0;
          stack[sp++] = c1;
        }
      } else if(hit0){
        stack[sp++] = c0;
      } else if(hit1){
        stack[sp++] = c1;
      }
    }
  }

  if(bestT < 1e19){
    vec3 hit = ro + bestT*rd;
    vec3 L   = normalize(light - hit);
    float d  = max(dot(bestN, L), 0.0);
    return vec3(d);
  }
  return vec3(0.6,0.8,1.0);
}

void main(){
  outColor = vec4(raytrace(v_uv),1.0);
}