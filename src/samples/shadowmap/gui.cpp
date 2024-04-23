#include "shadowmap_render.h"

#include "../../render/render_gui.h"

void SimpleShadowmapRender::SetupGUIElements()
{
  ImGui_ImplVulkan_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
  {
//    ImGui::ShowDemoWindow();
    ImGui::Begin("Simple render settings");

    ImGui::ColorEdit3("Meshes base color", m_uniforms.baseColor.M, ImGuiColorEditFlags_PickerHueWheel | ImGuiColorEditFlags_NoInputs);
    ImGui::SliderFloat3("Light source position", m_uniforms.lightPos.M, -10.f, 10.f);

    ImGui::SeparatorText("Screen-space ambient occlusion");

    ImGui::Checkbox("Use SSAO", &m_useSSAO);
    ImGui::ColorEdit3("Ambient color", m_ambientColor.M, ImGuiColorEditFlags_PickerHueWheel);
    ImGui::Checkbox("Use range check", &m_ssaoUseRangeCheck);
    ImGui::Checkbox("Use Gauss blur", &m_useBlur);
    ImGui::Checkbox("Use dir * ambient color", &m_useDirForAmbient);
    ImGui::SliderInt("Kernel size", (int*)&m_ssaoKernelSize, 1, 1024);
    ImGui::SliderFloat("Radius", &m_ssaoRadius, 0.1f, 1.0f);
    ImGui::SliderFloat("Bias", &m_ssaoBias, 0.001f, 0.01);

    ImGui::Separator();
    
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

    ImGui::NewLine();

    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f),"Press 'B' to recompile and reload shaders");
    ImGui::End();
  }

  // Rendering
  ImGui::Render();
}
