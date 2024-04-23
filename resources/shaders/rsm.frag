#version 450

#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

#include "common.h"

layout(location = 0) in f32vec3 in_ws_position;
layout(location = 1) in f32vec3 in_ws_normal;

layout(location = 0) out f32vec4 out_ws_position;
layout(location = 1) out f32vec4 out_ws_normal;
layout(location = 2) out f32vec4 out_radiant_flux;

layout(binding = 0, set = 0) uniform AppData {
	UniformParams params;
};

layout(push_constant) uniform PushContant {
    f32mat4 proj_view;
    f32mat4 model;
    f32vec3 base_color;
};

f32vec3 CalculateDirectLighting(f32vec3 ws_pos, f32vec3 ws_normal) {
	const f32vec3   color1      = f32vec3(0.5f, 0.5f, 0.5f);
	const f32vec3   color2      = f32vec3(1.0f, 1.0f, 1.0f);
	const f32vec3   light_color = mix(color1, color2, abs(sin(params.time)));
   
	const f32vec3   light_dir   = normalize(params.lightPos - ws_pos);
	const float32_t light_align = max(dot(ws_normal, light_dir), 0.0f);
  
	return light_color * light_align * base_color;
}

void main() {
	out_ws_position  = f32vec4(in_ws_position, 1.0f);
	out_ws_normal    = f32vec4(normalize(in_ws_normal), 1.0f);
	out_radiant_flux = f32vec4(CalculateDirectLighting(in_ws_position, normalize(in_ws_normal)), 1.0f);
}