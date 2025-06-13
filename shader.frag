#version 300 es
precision highp float;

layout(location = 0) out vec4 outColor;
in vec2 v_uv;

#define SPHERE_COUNT 10
#define SQRT_SAMPLES 1

uniform vec3 u_cameraPos;
uniform vec3 u_cameraTarget;
uniform vec3 u_cameraUp;
uniform float u_fov;
uniform float u_aspect;
uniform vec3  u_sphereCenters[SPHERE_COUNT];
uniform float u_sphereRadii[SPHERE_COUNT];
uniform vec3  u_colors[SPHERE_COUNT];
uniform float u_time;
uniform sampler2D u_prevFrameTex;
uniform int     u_frameIndex;
uniform int     u_maxFrames;

float sphereIntersect(vec3 ro, vec3 rd, vec3 c, float r) {
  vec3 oc = ro - c;
  float b = dot(oc, rd);
  float h = b*b - dot(oc, oc) + r*r;
  if (h < 0.0) return -1.0;
  return -b - sqrt(h);
}

float random(vec2 seed) {
  return fract(sin(dot(seed,vec2(12.9898,78.233)))*43758.5453123);
}

float computeSoftShadow(vec3 hit, vec3 normal, vec3 lightPos) {
  // compute basic ray & offset
  vec3 Ldir  = normalize(lightPos - hit);
  float Ldis = length(lightPos - hit);
  vec3 o = hit + normal * 0.001;

  // LOD parameters
  float nearDist = 5.0, farDist = 15.0;
  int minGrid = 1, maxGrid = 6;

  float d = clamp((Ldis - nearDist)/(farDist - nearDist),0.,1.);
  float q = pow(smoothstep(0.,1.,1.-d), 0.7);

  int S = max(int(mix(float(minGrid), float(maxGrid), q)+0.5),1);
  float invS = 1.0/float(S);

  // per-pixel rotation
  float angle = random(v_uv*37.0 + u_time*13.1) * 6.2831853;
  mat2 R = mat2(cos(angle),-sin(angle), sin(angle),cos(angle));

  float shadow = 0.0;
  for(int xi=0; xi<S; xi++){
    for(int yi=0; yi<S; yi++){
      vec2 jit = (vec2(xi,yi) + random(vec2(xi,yi)+v_uv))*invS - 0.5;
      jit = R*jit;
      vec3 Rd = normalize(Ldir + vec3(jit*0.05,0.0));
      bool blk=false;
      for(int j=0;j<SPHERE_COUNT;j++){
        float t=sphereIntersect(o,Rd,u_sphereCenters[j],u_sphereRadii[j]);
        if(t>0.0 && t< Ldis){ blk=true; break; }
      }
      shadow += blk?0.0:1.0;
    }
  }
  return shadow/float(S*S);
}

vec3 raytrace(vec2 uv) {
  vec3 fwd = normalize(u_cameraTarget - u_cameraPos);
  vec3 right=normalize(cross(fwd,u_cameraUp));
  vec3 upv =cross(right,fwd);

  uv = uv*2.0 -1.0;
  uv.x *= u_aspect;
  float focal = 1.0/tan(u_fov*0.5);
  vec3 ro = u_cameraPos;
  vec3 rd = normalize(fwd*focal + right*uv.x + upv*uv.y);

  float closest=1e20; int hit=-1;
  for(int i=0;i<SPHERE_COUNT;i++){
    float t=sphereIntersect(ro,rd,u_sphereCenters[i],u_sphereRadii[i]);
    if(t>0.0 && t<closest){ closest=t; hit=i; }
  }
  if(hit<0) return vec3(0.6,0.8,1.0);

  vec3 H = ro + closest*rd;
  vec3 N = normalize(H - u_sphereCenters[hit]);
  vec3 Lp=vec3(5.,5.,5.), Ld=normalize(Lp-H);
  float diff = max(dot(N,Ld),0.0);
  if(diff<=0.) return vec3(0.);

  float sh = computeSoftShadow(H,N,Lp);
  return u_colors[hit]*(diff*sh);
}

void main(){
  vec3 col=vec3(0.0);
  float offs=1.0/800.0;
  for(int dx=0;dx<SQRT_SAMPLES;dx++){
    for(int dy=0;dy<SQRT_SAMPLES;dy++){
      vec2 jit=(vec2(dx,dy)+random(v_uv+vec2(dx,dy)))/float(SQRT_SAMPLES);
      vec2 uv2=v_uv+(jit-0.5)*offs;
      col += raytrace(uv2);
    }
  }
  col /= float(SQRT_SAMPLES*SQRT_SAMPLES);

  vec4 curr = vec4(col,1.0);
  vec4 prev = texture(u_prevFrameTex, v_uv);
  float count= float(min(u_frameIndex+1, u_maxFrames));
  float alpha = 1.0/count;
  outColor = mix(prev, curr, alpha);
}