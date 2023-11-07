#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "common.h"
#include "unpack_attributes.h"

layout(location = 0) in vec4 vPosNorm;
layout(location = 1) in vec4 vTexCoordAndTang;

layout(push_constant) uniform params_t
{
    mat4 mProjView;
} params;


layout (location = 0 ) out VS_OUT
{
    vec3 wPos;
    vec3 wNorm;
    vec3 wTangent;
    vec2 texCoord;

} vOut;

layout(binding = 0) uniform AppData
{
    UniformParams Params;
};

layout(std430, binding = 1) readonly buffer Transforms {
    mat4 instance_transform[];
};

layout(std430, binding = 2) readonly buffer VisibleInstanceIndices {
    uint culled_instance_ids[];
};

out gl_PerVertex { vec4 gl_Position; };
void main(void)
{
    mat4 model = instance_transform[culled_instance_ids[gl_InstanceIndex]];
    model[3][1] += 0.7 * sin(culled_instance_ids[gl_InstanceIndex] + Params.time);

    const vec4 wNorm = vec4(DecodeNormal(floatBitsToInt(vPosNorm.w)),         0.0f);
    const vec4 wTang = vec4(DecodeNormal(floatBitsToInt(vTexCoordAndTang.z)), 0.0f);

    vOut.wPos     = (model * vec4(vPosNorm.xyz, 1.0f)).xyz;
    vOut.wNorm    = normalize(mat3(transpose(inverse(model))) * wNorm.xyz);
    vOut.wTangent = normalize(mat3(transpose(inverse(model))) * wTang.xyz);
    vOut.texCoord = vTexCoordAndTang.xy;

    gl_Position   = params.mProjView * vec4(vOut.wPos, 1.0);
}
