#version 450

#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

#include "common.h"

layout(location = 0) out vec4 out_color;

layout(location = 0) in TES_OUT {
    f32vec3 pos;
    f32vec3 norm;
} in_vertex;

layout(binding = 0, set = 0) uniform AppData {
    UniformParams Params;
};

void main() {
    const vec4 dark_violet = vec4(0.59f, 0.0f, 0.82f, 1.0f);
    const vec4 chartreuse  = vec4(0.5f, 1.0f, 0.0f, 1.0f);
    vec4       lightColor1 = mix(dark_violet, chartreuse, abs(sin(Params.time)));
    vec3       lightDir    = normalize(Params.lightPos - in_vertex.pos);
    vec4       lightColor  = max(dot(in_vertex.norm, lightDir), 0.0f) * lightColor1;
    out_color              = (lightColor + vec4(0.1f)) * vec4(Params.baseColor, 1.0f);
}
