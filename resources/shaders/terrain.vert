#version 450

#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

const f32vec3 QUAD_POSITIONS[4U] = f32vec3[] (
    f32vec3(-0.5f, 0.0f, -0.5f),
    f32vec3( 0.5f, 0.0f, -0.5f),
    f32vec3(-0.5f, 0.0f,  0.5f),
    f32vec3( 0.5f, 0.0f,  0.5f)
);

void main() {
    gl_Position = f32vec4(QUAD_POSITIONS[gl_VertexIndex], 1.0f);
}
