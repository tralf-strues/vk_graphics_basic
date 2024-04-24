#version 450

#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

layout(push_constant) uniform PushConstant {
    f32mat4  proj_view;
    f32mat4  model;
    f32vec2  height_range;
    uint32_t subdivisions;
} params;

layout (vertices = 4) out;
void main() {
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;

    for (uint32_t i = 0U; i < 4U; ++i) {
        gl_TessLevelOuter[i] = params.subdivisions;
    }

    for (uint32_t i = 0U; i < 2U; ++i) {
        gl_TessLevelInner[i] = params.subdivisions;
    }
    
}