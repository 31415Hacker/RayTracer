#version 300 es
precision highp float;
out vec4 outColor;
in vec2 v_uv;

#define SPHERE_COUNT 10
#define SQRT_SAMPLES 2

uniform vec3 u_cameraPos;
uniform vec3 u_cameraTarget;
uniform vec3 u_cameraUp;
uniform float u_fov;
uniform float u_aspect;
uniform vec3 u_sphereCenters[SPHERE_COUNT];
uniform float u_sphereRadii[SPHERE_COUNT];
uniform vec3 u_colors[SPHERE_COUNT];

float sphereIntersect(vec3 ro, vec3 rd, vec3 center, float radius) {
  vec3 oc = ro - center;
  float b = dot(oc, rd);
  float c = dot(oc, oc) - radius * radius;
  float h = b * b - c;
  if (h < 0.0) return -1.0;
  return -b - sqrt(h);
}

float random(vec2 seed) {
  return fract(sin(dot(seed.xy, vec2(12.9898,78.233))) * 43758.5453123);
}

float computeSoftShadow(vec3 hit, vec3 normal, vec3 lightPos) {
  vec3 lightDir   = normalize(lightPos - hit);
  float lightDist = length(lightPos - hit);
  vec3 o          = hit + normal * 0.001;

  // LOD as before…
  float nearDist = 1.0, farDist = 20.0;
  float q = clamp(1.0 - (lightDist - nearDist)/(farDist - nearDist), 0.0, 1.0);
  int minSamples = 1, maxSamples = 10;
  int totalSamples = int(mix(float(minSamples), float(maxSamples), q));
  int S = max(int(floor(sqrt(float(totalSamples)))), 1);
  float invS = 1.0/float(S);

  // --- compute a per-pixel rotation angle ---
  float angle = random(v_uv * 37.0) * 6.2831853; // [0,2π)
  mat2 R = mat2(
    cos(angle), -sin(angle),
    sin(angle),  cos(angle)
  );

  float shadow = 0.0;
  for (int xi = 0; xi < S; xi++) {
    for (int yi = 0; yi < S; yi++) {
      // stratified jitter in [−0.5,0.5]
      vec2 jitter = (vec2(xi, yi) + random(vec2(xi, yi) + v_uv)) * invS - 0.5;
      // rotate each sample
      jitter = R * jitter;

      vec3 offDir = normalize(lightDir + vec3(jitter * 0.05, 0.0));

      bool blocked = false;
      for (int j = 0; j < SPHERE_COUNT; j++) {
        float t = sphereIntersect(o, offDir, u_sphereCenters[j], u_sphereRadii[j]);
        if (t > 0.0 && t < lightDist) { blocked = true; break; }
      }
      shadow += blocked ? 0.0 : 1.0;
    }
  }
  return shadow / float(S * S);
}

vec3 raytrace(vec2 uv) {
  vec3 forward = normalize(u_cameraTarget - u_cameraPos);
  vec3 right = normalize(cross(forward, u_cameraUp));
  vec3 up = cross(right, forward);

  uv = uv * 2.0 - 1.0;
  uv.x *= u_aspect;
  float focalLength = 1.0 / tan(u_fov * 0.5);
  vec3 ro = u_cameraPos;
  vec3 rd = normalize(forward * focalLength + right * uv.x + up * uv.y);

  float closestT = 1e20;
  int hitSphere = -1;

  for (int i = 0; i < SPHERE_COUNT; i++) {
    float t = sphereIntersect(ro, rd, u_sphereCenters[i], u_sphereRadii[i]);
    if (t > 0.0 && t < closestT) {
      closestT = t;
      hitSphere = i;
    }
  }

  if (hitSphere != -1) {
    vec3 hit = ro + closestT * rd;
    vec3 normal = normalize(hit - u_sphereCenters[hitSphere]);

    vec3 lightPos = vec3(5.0, 5.0, 5.0);
    vec3 lightDir = normalize(lightPos - hit);

    float diffuse = max(dot(normal, lightDir), 0.0);

    // EARLY OUT: if no direct light, skip shadow entirely
    if (diffuse <= 0.0) {
      return vec3(0.0);  // or your ambient color if you have one
    }

    float shadow = computeSoftShadow(hit, normal, lightPos);

    vec3 color = u_colors[hitSphere];
    return color * (shadow * diffuse); // No ambient — pure soft shadow * diffuse
  }

  // Background
  return vec3(0.6, 0.8, 1.0);
}

void main() {
  vec3 color = vec3(0.0);
  float offset = 1.0 / 800.0;

  for (int dx = 0; dx < SQRT_SAMPLES; dx++) {
    for (int dy = 0; dy < SQRT_SAMPLES; dy++) {
      // stratified offset in [0,1)
      vec2 jitter = (vec2(dx, dy) + random(v_uv + vec2(dx,dy))) / float(SQRT_SAMPLES);
      // center it around zero
      vec2 offUV = v_uv + (jitter - 0.5) * offset;

      color += raytrace(offUV);
    }
  }

  color /= float(SQRT_SAMPLES * SQRT_SAMPLES);
  outColor = vec4(color, 1.0);
}