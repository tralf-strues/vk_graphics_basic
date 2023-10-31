#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

#include "unpack_attributes.h"
#include "common.h"

layout(location = 0) in vec4 vPosNorm;
layout(location = 1) in vec4 vTexCoordAndTang;

layout(push_constant) uniform params_t
{
    mat4 mProjView;
    mat4 mModel;
} params;

layout(binding = 0, set = 0) uniform AppData
{
    UniformParams Params;
};

layout (location = 0 ) out VS_OUT
{
    vec3 wPos;
    vec3 wNorm;
    vec3 wTangent;
    vec2 texCoord;

} vOut;

mat3 RotateY(float theta) {
    float c = cos(theta);
    float s = sin(theta);
    return mat3(
        vec3(c, 0, s),
        vec3(0, 1, 0),
        vec3(-s, 0, c)
    );
}

float Random2d(in vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898,78.233))) * 43758.5453123);
}

// 2D Noise based on Morgan McGuire @morgan3d
// https://www.shadertoy.com/view/4dS3Wd
float Noise(in vec2 st) {
    vec2 i = floor(st);
    vec2 f = fract(st);

    float a = Random2d(i);
    float b = Random2d(i + vec2(1.0, 0.0));
    float c = Random2d(i + vec2(0.0, 1.0));
    float d = Random2d(i + vec2(1.0, 1.0));

    vec2 u = f*f*(3.0-2.0*f);

    return mix(a, b, u.x) +
            (c - a)* u.y * (1.0 - u.x) +
            (d - b) * u.x * u.y;
}

void AnimationNoize(inout vec3 mPos, in vec3 mNorm, in vec2 texCoord, in vec2 dir, in float blend) {
    const vec2 texOffset = dir * Params.time;

    const float displacement = Noise(8.0 * (texCoord + texOffset));
    const float coeff = 0.02;

    mPos += blend * coeff * normalize(mNorm) * displacement;
}

void AnimationDisplace(inout vec3 mPos, in vec3 mNorm, in float final, in float t) {
    mPos += normalize(mNorm) * mix(0, final, t);
}

void AnimationScale(inout vec3 mPos, in float start, in float end, in float t) {
    vec3 startScaled = mPos * start;
    vec3 endScaled = mPos * end;
    mPos = mix(startScaled, endScaled, t);
}

void AnimationRotate(inout vec3 mPos, in float start, in float end, in float t) {
    mPos = RotateY(mix(start, end, t)) * mPos;
}

void AnimationMove(inout vec3 mPos, in vec3 mFinalMove, in float t) {
    mPos = mix(mPos, mPos + mFinalMove, t);
}

#define PI 3.1415926

void Animation(inout vec3 mPos, in vec3 mNorm, in vec2 texCoord) {
    const float TIME_0 = 3.0;
    const float TIME_1 = 1.0;
    const float TIME_2 = 1.0;

    const float TOTAL_TIME = TIME_0 + TIME_1 + TIME_2;
    float time = Params.time - (TOTAL_TIME * floor(Params.time/TOTAL_TIME));
    
    if (time <= TIME_0) {
        float t = time / TIME_0;
        AnimationNoize(mPos, mNorm, texCoord, vec2(0.5,  0.5), 1.0);
        AnimationNoize(mPos, mNorm, texCoord, vec2(0.5, -0.5), 1.0);
        AnimationScale(mPos, 1.0, 0.5, t);
    } else if (time <= TIME_0 + TIME_1) {
        float t = (time - TIME_0) / TIME_1;

        AnimationScale(mPos, 0.5, 0.1, t);
        AnimationRotate(mPos, 0, 50.0 * 2 * PI, t);
        AnimationMove(mPos, vec3(0.0, -2.0, 0.0), t);
    } else {
        float t = (time - (TIME_0 + TIME_1)) / TIME_2;
        AnimationScale(mPos, 0.1, 0.5, t);
        AnimationMove(mPos, vec3(0.0, -2.0, 0.0), 1.0);
        AnimationDisplace(mPos, mNorm, 1.1, t);
        AnimationRotate(mPos, 0, 3 * 2 * PI, t);
    }
}

out gl_PerVertex { vec4 gl_Position; };
void main(void)
{
    const vec3 mNorm = DecodeNormal(floatBitsToInt(vPosNorm.w));
    const vec3 mTang = DecodeNormal(floatBitsToInt(vTexCoordAndTang.z));
    const vec2 texCoord = vTexCoordAndTang.xy;
    vec3 mPos = vPosNorm.xyz;
    
    Animation(mPos, mNorm, texCoord);

    vOut.wPos     = (params.mModel * vec4(mPos, 1.0f)).xyz;
    vOut.wNorm    = normalize(mat3(transpose(inverse(params.mModel))) * mNorm);
    vOut.wTangent = normalize(mat3(transpose(inverse(params.mModel))) * mTang);
    vOut.texCoord = texCoord;

    gl_Position   = params.mProjView * vec4(vOut.wPos, 1.0);
}
