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

    ImGui::SeparatorText("Heightmap");
    ImGui::SliderInt("Num octaves", (int*)&m_heightmapParams.octavesCount, 1, 16);
    ImGui::SliderFloat("Base frequency", &m_heightmapParams.baseFrequency, 0.1, 4.0f, "%.2f");
    
    ImGui::NewLine();
    ImGui::Text("Smoothing: pow(fudge * h, exponent)");
    ImGui::SliderFloat("Fudge", &m_heightmapParams.fudgeFactor, 0.8f, 1.2f, "%.2f");
    ImGui::SliderFloat("Exponent", &m_heightmapParams.heightExponent, 0.01f, 10.0f, "%.2f");
    ImGui::NewLine();

    ImGui::Separator();

    ImGui::NewLine();
    ImGui::SeparatorText("Tessellation");

    ImGui::SliderFloat2("Terrain scale", &m_terrainScale.x, 0.1f, 10.f);
    ImGui::SliderFloat("Terrain y", &m_terrainY, -10.f, 10.f);
    ImGui::SliderFloat2("Height range", &m_terrainParams.heightRange.x, -1.0f, 1.0f, "%.01f");
    ImGui::SliderInt("Num subdivisions", (int*)&m_terrainParams.subdivisions, 4, 64);
    
    ImGui::NewLine();
    ImGui::Separator();
    ImGui::NewLine();

    ImGui::NewLine();
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::NewLine();

    ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f),"Press 'B' to recompile and reload shaders");
    ImGui::End();
  }

  // Rendering
  ImGui::Render();
}
