#version 450

#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

#include "unpack_attributes.h"

layout(location = 0) in f32vec4 in_pos_normal;
layout(location = 1) in f32vec4 in_tex_coords_tangant;

layout(push_constant) uniform PushContant {
    f32mat4 proj_view;
    f32mat4 model;
};

layout(location = 0) out f32vec3 out_ws_position;
layout(location = 1) out f32vec3 out_ws_normal;

void main(void)
{
    const f32vec4 ms_normal = f32vec4(DecodeNormal(floatBitsToInt(in_pos_normal.w)), 0.0f);

    out_ws_position         = (model * f32vec4(in_pos_normal.xyz, 1.0f)).xyz;
    out_ws_normal           = normalize(f32mat3(transpose(inverse(model))) * ms_normal.xyz);

    gl_Position             = proj_view * f32vec4(out_ws_position, 1.0f);
}
