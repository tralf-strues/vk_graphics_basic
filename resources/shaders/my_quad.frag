#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) out vec4 color;

layout (binding = 0) uniform sampler2D colorTex;

layout (location = 0 ) in VS_OUT
{
  vec2 texCoord;
} surf;

const vec3 LUMA_COEFFS = vec3(0.2126, 0.7152, 0.0722);

float CompareLuma(in vec4 first, in vec4 second) {
  float luma_first  = dot(vec3(first),  LUMA_COEFFS);
  float luma_second = dot(vec3(second), LUMA_COEFFS);

  return luma_first - luma_second;
}

void BubbleSort9(inout vec4 samples[9]) {
  for (int i = 0; i < 9; i++) {
    int swapped = 0;

    for (int j = 0; j < 9 - i; j++) {
      if (CompareLuma(samples[j], samples[j + 1]) > 0.0) {
        vec4 tmp = samples[j];
        samples[j]     = samples[j + 1];
        samples[j + 1] = tmp;

        swapped = 1;
      }
    }

    if (swapped == 0) {
      break;
    }
  }
}

void main()
{
  vec4 samples[9] = vec4[9](
    textureOffset(colorTex, surf.texCoord, ivec2(-1, -1)),
    textureOffset(colorTex, surf.texCoord, ivec2(-1,  0)),
    textureOffset(colorTex, surf.texCoord, ivec2(-1,  1)),
    textureOffset(colorTex, surf.texCoord, ivec2( 0, -1)),
    textureOffset(colorTex, surf.texCoord, ivec2( 0,  0)),
    textureOffset(colorTex, surf.texCoord, ivec2( 0,  1)),
    textureOffset(colorTex, surf.texCoord, ivec2( 1, -1)),
    textureOffset(colorTex, surf.texCoord, ivec2( 1,  0)),
    textureOffset(colorTex, surf.texCoord, ivec2( 1,  1))
  );

  BubbleSort9(samples);

  color = samples[4];
  //color = texture(colorTex, surf.texCoord);
}
