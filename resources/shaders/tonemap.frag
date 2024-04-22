#version 450

layout (binding = 0) uniform sampler2D original_image;

layout(push_constant) uniform params_t
{
    float gamma;
    float exposure;
    bool  use_tonemapping;
};

layout (location = 0 ) in vec2 in_uv;

layout (location = 0 ) out vec4 out_color;

void main() {
    vec3 hdr_color = texture(original_image, in_uv).rgb;
    
    if (use_tonemapping) {
        vec3 ldr_color       = vec3(1.0f) - exp(-hdr_color * exposure);
        vec3 gamma_corrected = pow(ldr_color, vec3(1.0f / gamma));
  
        out_color = vec4(gamma_corrected, 1.0f);
    } else {
        out_color = vec4(min(hdr_color, vec3(1.0f)), 1.0f);
    }
}