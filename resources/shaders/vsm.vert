#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "unpack_attributes.h"

layout(location = 0) in vec4 vPosNorm;
layout(location = 1) in vec4 vTexCoordAndTang;

layout(push_constant) uniform params_t
{
    mat4 mProjView;
    mat4 mModel;
} params;

out gl_PerVertex { vec4 gl_Position; };
void main(void)
{
    const vec3 wPos = (params.mModel * vec4(vPosNorm.xyz, 1.0f)).xyz;
    gl_Position     = params.mProjView * vec4(wPos, 1.0);
}
