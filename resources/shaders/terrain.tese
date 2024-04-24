#version 450

#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

layout(set = 0, binding = 1) uniform sampler2D height_map;

layout(push_constant) uniform PushConstant {
    f32mat4  proj_view;
    f32mat4  model;
    f32vec2  height_range;
    uint32_t subdivisions;
} params;

layout(location = 0) out TES_OUT {
    f32vec3 pos;
    f32vec3 norm;
} out_vertex;

float32_t SampleHeight(f32vec2 uv) {
    float height = texture(height_map, uv).x;
    height = mix(params.height_range.x, params.height_range.y, height);

    return height;
}

f32vec3 CalculateNormal(f32vec2 uv) {
    const float32_t SAMPLE_STEP = 4.0f / params.subdivisions;

    const f32vec2 cross_uv[4U] = f32vec2[](
        uv - f32vec2(SAMPLE_STEP,        0.0f), // 0: left
        uv + f32vec2(SAMPLE_STEP,        0.0f), // 1: right
        uv - f32vec2(       0.0f, SAMPLE_STEP), // 2: bottom
        uv + f32vec2(       0.0f, SAMPLE_STEP)  // 3: top
    );

    f32vec3 normal = f32vec3(0.0f, 1.0f, 0.0f);
    normal.x = SampleHeight(cross_uv[1U]) - SampleHeight(cross_uv[0U]);
    normal.z = SampleHeight(cross_uv[3U]) - SampleHeight(cross_uv[2U]);

    normal.xz /= 2.0f * SAMPLE_STEP;

    return normalize(normal);
}

layout(quads, equal_spacing, ccw) in;
void main() {
    f32vec3 quad_pos[4U];
    for (uint32_t i = 0U; i < 4U; ++i) {
        quad_pos[i] = gl_in[i].gl_Position.xyz;
    }

    f32vec2   uv        = gl_TessCoord.xy;
    float32_t height    = SampleHeight(uv);
    f32vec3   ms_pos    = f32vec3(0.0f, height, 0.0f) + mix(quad_pos[0U], quad_pos[1U], uv.x) + mix(quad_pos[0U], quad_pos[3U], uv.y);
    f32vec3   ms_normal = CalculateNormal(uv);

    out_vertex.pos      = (params.model * f32vec4(ms_pos, 1.0f)).xyz;
    out_vertex.norm     = normalize(f32mat3(transpose(inverse(params.model))) * ms_normal);
    gl_Position         = params.proj_view * f32vec4(out_vertex.pos, 1.0f);
}