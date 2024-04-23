#version 450

#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

#include "common.h"
#include "rsm_samples.glsl"

layout(location = 0) in f32vec3 in_ws_position;
layout(location = 1) in f32vec3 in_ws_normal;
layout(location = 2) in f32vec3 in_ws_tangent;
layout(location = 3) in f32vec2 in_tex_coords;

layout(location = 0) out f32vec4 out_color;

layout(push_constant) uniform PushContant {
    f32mat4 proj_view;
    f32mat4 model;
    f32vec3 base_color;
};

layout(set = 0, binding = 0) uniform AppData {
    UniformParams params;
};

layout(set = 0, binding = 1) uniform sampler2D rsm_depth;
layout(set = 0, binding = 2) uniform sampler2D rsm_position;
layout(set = 0, binding = 3) uniform sampler2D rsm_normal;
layout(set = 0, binding = 4) uniform sampler2D rsm_flux;

f32vec3 ConvertToShadowMapCoords(f32vec3 ws_pos) {
    const f32vec4 pos_light_cs  = params.lightMatrix * f32vec4(ws_pos, 1.0f);
    const f32vec3 pos_light_ndc = pos_light_cs.xyz / pos_light_cs.w;              // for orto matrix, we don't need perspective division, you can remove it if you want; this is general case;
    const f32vec2 shadow_uv     = pos_light_ndc.xy * 0.5f + f32vec2(0.5f, 0.5f);  // just shift coords from [-1,1] to [0,1]

    return f32vec3(shadow_uv, pos_light_ndc.z);
}

f32vec3 CalculateDirectLighting(f32vec3 ws_pos, f32vec3 ws_normal, float32_t shadow) {
	const f32vec3   color1      = f32vec3(0.5f, 0.5f, 0.5f);
	const f32vec3   color2      = f32vec3(1.0f, 1.0f, 1.0f);
	const f32vec3   light_color = mix(color1, color2, abs(sin(params.time)));
   
	const f32vec3   light_dir   = normalize(params.lightPos - ws_pos);
	const float32_t light_align = max(dot(ws_normal, light_dir), 0.0f);
  
	return (light_color * light_align * shadow + vec3(0.1f)) * base_color;
}

struct SampleData {
    f32vec3 ws_pos;
    f32vec3 ws_normal;
    f32vec3 flux;
};

void GetSampleData(out SampleData sample_data, f32vec2 coords) {
    sample_data.ws_pos    = texture(rsm_position, coords).xyz;
    sample_data.ws_normal = texture(rsm_normal,   coords).xyz;
    sample_data.flux      = texture(rsm_flux,     coords).rgb;   
}

f32vec3 CalculateIndirectLighting(f32vec3 ws_pos, f32vec3 ws_normal) {
    f32vec3 shadow_coords  = ConvertToShadowMapCoords(ws_pos);
    f32vec3 indirect_light = f32vec3(0.0f);

    SampleData sample_data;
    for (uint32_t i = 0u; i < RSM_SAMLE_COUNT; ++i) {
        f32vec2 rnd_coords = RSM_SAMPLE_COORDS[i];
        f32vec2 coords     = shadow_coords.xy + params.rsm_rmax * rnd_coords;

        GetSampleData(sample_data, coords);

        float32_t dist  = max(0.1f, length(ws_pos - sample_data.ws_pos));
        float32_t dist4 = dist * dist * dist * dist;
        
        f32vec3 sample_radiance = sample_data.flux *
                                  max(0.0f, dot(sample_data.ws_normal, ws_pos - sample_data.ws_pos)) *
                                  max(0.0f, dot(ws_normal, sample_data.ws_pos - ws_pos)) /
                                  dist4;

        float32_t sample_weight = rnd_coords.x * rnd_coords.x;
        indirect_light += sample_radiance * sample_weight;
    }

    return clamp(indirect_light * params.rsm_intensity, 0.0f, 1.0f);
}

void main() {
    const f32vec3   shadow_coords  = ConvertToShadowMapCoords(in_ws_position);

    const bool      out_of_view    = (shadow_coords.x < 0.0001f || shadow_coords.x > 0.9999f || shadow_coords.y < 0.0091f || shadow_coords.y > 0.9999f);
    const float32_t shadow         = ((shadow_coords.z < textureLod(rsm_depth, shadow_coords.xy, 0.0f).x + 0.001f) || out_of_view) ? 1.0f : 0.0f;

    const f32vec3   direct_light   = CalculateDirectLighting(in_ws_position, in_ws_normal, shadow);
    const f32vec3   indirect_light = CalculateIndirectLighting(in_ws_position, in_ws_normal);
    out_color                      = f32vec4(direct_light + indirect_light, 1.0f);
}
