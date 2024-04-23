#include "shadowmap_render.h"

#include <geom/vk_mesh.h>
#include <vk_pipeline.h>
#include <vk_buffers.h>
#include <iostream>
#include <random>

#include <etna/GlobalContext.hpp>
#include <etna/Etna.hpp>
#include <etna/RenderTargetStates.hpp>
#include <vulkan/vulkan_core.h>

constexpr vk::Format SSAO_IMAGE_FORMAT    = vk::Format::eR16Unorm;
constexpr uint32_t   SSAO_KERNEL_MAX_SIZE = 1024U;

float RandomInRange(float start, float end) {
  static std::random_device device;
  static std::mt19937       generator(device());

  std::uniform_real_distribution<float> dist(start, end);
  return dist(generator);
}

/// RESOURCE ALLOCATION

void SimpleShadowmapRender::AllocateResources()
{
  mainViewDepth = m_context->createImage(etna::Image::CreateInfo
  {
    .extent = vk::Extent3D{m_width, m_height, 1},
    .name = "main_view_depth",
    .format = vk::Format::eD32Sfloat,
    .imageUsage = vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled 
  });

  shadowMap = m_context->createImage(etna::Image::CreateInfo
  {
    .extent = vk::Extent3D{2048, 2048, 1},
    .name = "shadow_map",
    .format = vk::Format::eD16Unorm,
    .imageUsage = vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled
  });

  m_ssaoImage = m_context->createImage(etna::Image::CreateInfo
  {
    .extent = vk::Extent3D{m_width, m_height, 1},
    .name = "ssao",
    .format = SSAO_IMAGE_FORMAT,
    .imageUsage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled
  });

  m_ssaoBlurTempImage = m_context->createImage(etna::Image::CreateInfo
  {
    .extent = vk::Extent3D{m_width, m_height, 1},
    .name = "ssao_blur_temp",
    .format = SSAO_IMAGE_FORMAT,
    .imageUsage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled
  });

  m_ssaoKernelBuffer = m_context->createBuffer(etna::Buffer::CreateInfo
  {
    .size = sizeof(float4) * SSAO_KERNEL_MAX_SIZE,
    .bufferUsage = vk::BufferUsageFlagBits::eStorageBuffer,
    .memoryUsage = VMA_MEMORY_USAGE_CPU_ONLY,
    .name = "ssao_kernel"
  });

  defaultSampler = etna::Sampler(etna::Sampler::CreateInfo{.name = "default_sampler"});
  constants = m_context->createBuffer(etna::Buffer::CreateInfo
  {
    .size = sizeof(UniformParams),
    .bufferUsage = vk::BufferUsageFlagBits::eUniformBuffer,
    .memoryUsage = VMA_MEMORY_USAGE_CPU_ONLY,
    .name = "constants"
  });

  m_uboMappedMem = constants.map();
  m_ssaoKernelMappedMemory = m_ssaoKernelBuffer.map();
}

void SimpleShadowmapRender::LoadScene(const char* path, bool transpose_inst_matrices)
{
  m_pScnMgr->LoadSceneXML(path, transpose_inst_matrices);

  // TODO: Make a separate stage
  loadShaders();
  PreparePipelines();

  auto loadedCam = m_pScnMgr->GetCamera(0);
  m_cam.fov = loadedCam.fov;
  m_cam.pos = float3(loadedCam.pos);
  m_cam.up  = float3(loadedCam.up);
  m_cam.lookAt = float3(loadedCam.lookAt);
  m_cam.tdist  = loadedCam.farPlane;
}

void SimpleShadowmapRender::DeallocateResources()
{
  mainViewDepth.reset(); // TODO: Make an etna method to reset all the resources
  shadowMap.reset();
  m_ssaoImage.reset();
  m_ssaoBlurTempImage.reset();
  m_ssaoKernelBuffer.reset();
  m_swapchain.Cleanup();
  vkDestroySurfaceKHR(GetVkInstance(), m_surface, nullptr);  

  constants = etna::Buffer();
}


/// PIPELINES CREATION

void SimpleShadowmapRender::PreparePipelines()
{
  // create full screen quad for debug purposes
  // 
  m_pQuad = std::make_unique<QuadRenderer>(QuadRenderer::CreateInfo{ 
      .format = static_cast<vk::Format>(m_swapchain.GetFormat()),
      .rect = { 0, 0, 512, 512 }, 
    });
  SetupSimplePipeline();
}

void SimpleShadowmapRender::loadShaders()
{
  etna::create_program("simple_material", {VK_GRAPHICS_BASIC_ROOT"/resources/shaders/simple_shadow.frag.spv", VK_GRAPHICS_BASIC_ROOT"/resources/shaders/simple.vert.spv"});
  etna::create_program("simple_shadow", {VK_GRAPHICS_BASIC_ROOT"/resources/shaders/simple.vert.spv"});
  etna::create_program("depth_prepass", {VK_GRAPHICS_BASIC_ROOT"/resources/shaders/simple.vert.spv"});
  etna::create_program("ssao", {VK_GRAPHICS_BASIC_ROOT"/resources/shaders/ssao.comp.spv"});
  etna::create_program("ssao_blur", { VK_GRAPHICS_BASIC_ROOT "/resources/shaders/ssao_blur.comp.spv"});
  etna::create_program("forward_ssao", {VK_GRAPHICS_BASIC_ROOT"/resources/shaders/simple_shadow_ssao.frag.spv", VK_GRAPHICS_BASIC_ROOT"/resources/shaders/simple.vert.spv"});
}

void SimpleShadowmapRender::SetupSimplePipeline()
{
  etna::VertexShaderInputDescription sceneVertexInputDesc
    {
      .bindings = {etna::VertexShaderInputDescription::Binding
        {
          .byteStreamDescription = m_pScnMgr->GetVertexStreamDescription()
        }}
    };

  auto& pipelineManager = etna::get_context().getPipelineManager();
  m_basicForwardPipeline = pipelineManager.createGraphicsPipeline("simple_material",
    {
      .vertexShaderInput = sceneVertexInputDesc,
      .fragmentShaderOutput =
        {
          .colorAttachmentFormats = {static_cast<vk::Format>(m_swapchain.GetFormat())},
          .depthAttachmentFormat = vk::Format::eD32Sfloat
        }
    });
  m_shadowPipeline = pipelineManager.createGraphicsPipeline("simple_shadow",
    {
      .vertexShaderInput = sceneVertexInputDesc,
      .fragmentShaderOutput =
        {
          .depthAttachmentFormat = vk::Format::eD16Unorm
        }
    });
  m_depthPrepassPipeline = pipelineManager.createGraphicsPipeline("depth_prepass",
    {
      .vertexShaderInput = sceneVertexInputDesc,
      .fragmentShaderOutput =
        {
          .depthAttachmentFormat = vk::Format::eD32Sfloat
        }
    });
  m_ssaoPipeline = pipelineManager.createComputePipeline("ssao", {});
  m_ssaoBlurPipeline = pipelineManager.createComputePipeline("ssao_blur", {});
  m_forwardSSAOPipeline = pipelineManager.createGraphicsPipeline("forward_ssao",
    {
      .vertexShaderInput = sceneVertexInputDesc,
      .fragmentShaderOutput =
        {
          .colorAttachmentFormats = {static_cast<vk::Format>(m_swapchain.GetFormat())},
          .depthAttachmentFormat = vk::Format::eD32Sfloat
        }
    });
}

void SimpleShadowmapRender::RecalculateKernelSamples() {
  if (m_ssaoKernelSize != m_ssaoKernelSamples.size()) {
    m_ssaoKernelSamples.resize(m_ssaoKernelSize);

    for (uint32_t i = 0U; i < m_ssaoKernelSize; ++i) {
      float scale = float(i) / float(m_ssaoKernelSize);
      scale = lerp(0.1f, 1.0f, scale * scale);
      float3 sample = scale * normalize(float3(RandomInRange(-1.0f, 1.0f), RandomInRange(-1.0f, 1.0f), RandomInRange(0.0f, 1.0f)));

      m_ssaoKernelSamples[i] = float4(sample.x, sample.y, sample.z, 0.0f);
    }

    memcpy(m_ssaoKernelMappedMemory, m_ssaoKernelSamples.data(), m_ssaoKernelSize * sizeof(m_ssaoKernelSamples[0U]));
  }
}

/// COMMAND BUFFER FILLING

void SimpleShadowmapRender::DrawSceneCmd(VkCommandBuffer a_cmdBuff, const float4x4& a_wvp, VkPipelineLayout a_pipelineLayout)
{
  VkShaderStageFlags stageFlags = (VK_SHADER_STAGE_VERTEX_BIT);

  VkDeviceSize zero_offset = 0u;
  VkBuffer vertexBuf = m_pScnMgr->GetVertexBuffer();
  VkBuffer indexBuf  = m_pScnMgr->GetIndexBuffer();
  
  vkCmdBindVertexBuffers(a_cmdBuff, 0, 1, &vertexBuf, &zero_offset);
  vkCmdBindIndexBuffer(a_cmdBuff, indexBuf, 0, VK_INDEX_TYPE_UINT32);

  pushConst2M.projView = a_wvp;
  for (uint32_t i = 0; i < m_pScnMgr->InstancesNum(); ++i)
  {
    auto inst         = m_pScnMgr->GetInstanceInfo(i);
    pushConst2M.model = m_pScnMgr->GetInstanceMatrix(i);
    vkCmdPushConstants(a_cmdBuff, a_pipelineLayout,
      stageFlags, 0, sizeof(pushConst2M), &pushConst2M);

    auto mesh_info = m_pScnMgr->GetMeshInfo(inst.mesh_id);
    vkCmdDrawIndexed(a_cmdBuff, mesh_info.m_indNum, 1, mesh_info.m_indexOffset, mesh_info.m_vertexOffset, 0);
  }
}

void SimpleShadowmapRender::BuildCommandBufferSimple(VkCommandBuffer a_cmdBuff, VkImage a_targetImage, VkImageView a_targetImageView)
{
  vkResetCommandBuffer(a_cmdBuff, 0);

  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

  VK_CHECK_RESULT(vkBeginCommandBuffer(a_cmdBuff, &beginInfo));

  //// draw scene to shadowmap
  //
  {
    etna::RenderTargetState renderTargets(a_cmdBuff, {0, 0, 2048, 2048}, {}, {.image = shadowMap.get(), .view = shadowMap.getView({})});

    vkCmdBindPipeline(a_cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, m_shadowPipeline.getVkPipeline());
    DrawSceneCmd(a_cmdBuff, m_lightMatrix, m_shadowPipeline.getVkPipelineLayout());
  }

  //// draw final scene to screen
  //
  {
    auto simpleMaterialInfo = etna::get_shader_program("simple_material");

    auto set = etna::create_descriptor_set(simpleMaterialInfo.getDescriptorLayoutId(0), a_cmdBuff,
    {
      etna::Binding {0, constants.genBinding()},
      etna::Binding {1, shadowMap.genBinding(defaultSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)}
    });

    VkDescriptorSet vkSet = set.getVkSet();

    etna::RenderTargetState renderTargets(a_cmdBuff, {0, 0, m_width, m_height},
      {{.image = a_targetImage, .view = a_targetImageView}},
      {.image = mainViewDepth.get(), .view = mainViewDepth.getView({})});

    vkCmdBindPipeline(a_cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, m_basicForwardPipeline.getVkPipeline());
    vkCmdBindDescriptorSets(a_cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS,
      m_basicForwardPipeline.getVkPipelineLayout(), 0, 1, &vkSet, 0, VK_NULL_HANDLE);

    DrawSceneCmd(a_cmdBuff, m_worldViewProj, m_basicForwardPipeline.getVkPipelineLayout());
  }

  if(m_input.drawFSQuad)
    m_pQuad->RecordCommands(a_cmdBuff, a_targetImage, a_targetImageView, shadowMap, defaultSampler);

  etna::set_state(a_cmdBuff, a_targetImage, vk::PipelineStageFlagBits2::eBottomOfPipe,
    vk::AccessFlags2(), vk::ImageLayout::ePresentSrcKHR,
    vk::ImageAspectFlagBits::eColor);

  etna::finish_frame(a_cmdBuff);

  VK_CHECK_RESULT(vkEndCommandBuffer(a_cmdBuff));
}

void SimpleShadowmapRender::BuildCommandBufferSSAO(VkCommandBuffer a_cmdBuff, VkImage a_targetImage, VkImageView a_targetImageView)
{
  vkResetCommandBuffer(a_cmdBuff, 0);

  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

  VK_CHECK_RESULT(vkBeginCommandBuffer(a_cmdBuff, &beginInfo));

  //// draw scene to shadowmap
  //
  {
    etna::RenderTargetState renderTargets(a_cmdBuff, {0, 0, 2048, 2048}, {}, {.image = shadowMap.get(), .view = shadowMap.getView({})});

    vkCmdBindPipeline(a_cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, m_shadowPipeline.getVkPipeline());
    DrawSceneCmd(a_cmdBuff, m_lightMatrix, m_shadowPipeline.getVkPipelineLayout());
  }

  //// depth prepass
  //
  {
    etna::RenderTargetState renderTargets(a_cmdBuff, {0, 0, m_width, m_height}, {}, {.image = mainViewDepth.get(), .view = mainViewDepth.getView({})});

    vkCmdBindPipeline(a_cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, m_depthPrepassPipeline.getVkPipeline());
    DrawSceneCmd(a_cmdBuff, m_worldViewProj, m_depthPrepassPipeline.getVkPipelineLayout());
  }

  etna::set_state(a_cmdBuff, mainViewDepth.get(), vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead, vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageAspectFlagBits::eDepth);
  etna::set_state(a_cmdBuff, m_ssaoImage.get(), vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite, vk::ImageLayout::eGeneral, vk::ImageAspectFlagBits::eColor);
  etna::flush_barriers(a_cmdBuff);

  //// ssao
  //
  {
    auto ssao = etna::get_shader_program("ssao");

    auto set = etna::create_descriptor_set(ssao.getDescriptorLayoutId(0), a_cmdBuff, {
        etna::Binding{ 0, mainViewDepth.genBinding(defaultSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal) },
        etna::Binding{ 1, m_ssaoImage.genBinding(defaultSampler.get(), vk::ImageLayout::eGeneral) },
        etna::Binding{ 2, m_ssaoKernelBuffer.genBinding() }
    });

    VkDescriptorSet vkSet = set.getVkSet();

    vkCmdBindPipeline(a_cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, m_ssaoPipeline.getVkPipeline());
    vkCmdBindDescriptorSets(a_cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, m_ssaoPipeline.getVkPipelineLayout(), 0, 1, &vkSet, 0, VK_NULL_HANDLE);

    struct {
      float4x4 inv_proj_view;
      float4x4 proj_view;
      float2   inv_texture_size;
      uint32_t kernel_size;
      float    radius;
      float    bias;
      uint32_t use_range_check;
    } push_constant {
      .inv_proj_view    = inverse4x4(m_worldViewProj),
      .proj_view        = m_worldViewProj,
      .inv_texture_size = 1.0f / float2(m_width, m_height),
      .kernel_size      = m_ssaoKernelSize,
      .radius           = m_ssaoRadius,
      .bias             = m_ssaoBias,
      .use_range_check  = m_ssaoUseRangeCheck
    };

    vkCmdPushConstants(a_cmdBuff, m_ssaoPipeline.getVkPipelineLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constant), &push_constant);
    vkCmdDispatch(a_cmdBuff, (m_width + 7) / 8, (m_height + 7) / 8, 1);
  }

  etna::set_state(a_cmdBuff, m_ssaoImage.get(), vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead, vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageAspectFlagBits::eColor);
  etna::set_state(a_cmdBuff, m_ssaoBlurTempImage.get(), vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite, vk::ImageLayout::eGeneral, vk::ImageAspectFlagBits::eColor);
  etna::flush_barriers(a_cmdBuff);

  if (m_useBlur) {
    auto blur = etna::get_shader_program("ssao_blur");
    auto set = etna::create_descriptor_set(blur.getDescriptorLayoutId(0), a_cmdBuff, {
        etna::Binding{ 0, m_ssaoImage.genBinding(defaultSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal) },
        etna::Binding{ 1, m_ssaoBlurTempImage.genBinding(defaultSampler.get(), vk::ImageLayout::eGeneral) }
    });

    VkDescriptorSet vkSet = set.getVkSet();

    vkCmdBindPipeline(a_cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, m_ssaoBlurPipeline.getVkPipeline());
    vkCmdBindDescriptorSets(a_cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, m_ssaoBlurPipeline.getVkPipelineLayout(), 0, 1, &vkSet, 0, VK_NULL_HANDLE);
    float direction[2U] = {1.0f, 0.0f};
    vkCmdPushConstants(a_cmdBuff, m_ssaoBlurPipeline.getVkPipelineLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(direction), direction);
    vkCmdDispatch(a_cmdBuff, (m_width + 31) / 32, (m_height + 31) / 32, 1);

    etna::set_state(a_cmdBuff, m_ssaoBlurTempImage.get(), vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderRead, vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageAspectFlagBits::eColor);
    etna::set_state(a_cmdBuff, m_ssaoImage.get(), vk::PipelineStageFlagBits2::eComputeShader, vk::AccessFlagBits2::eShaderWrite, vk::ImageLayout::eGeneral, vk::ImageAspectFlagBits::eColor);
    etna::flush_barriers(a_cmdBuff);

    set = etna::create_descriptor_set(blur.getDescriptorLayoutId(0), a_cmdBuff, {
        etna::Binding{ 0, m_ssaoBlurTempImage.genBinding(defaultSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal) },
        etna::Binding{ 1, m_ssaoImage.genBinding(defaultSampler.get(), vk::ImageLayout::eGeneral) }
    });

    vkSet = set.getVkSet();
    vkCmdBindDescriptorSets(a_cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, m_ssaoBlurPipeline.getVkPipelineLayout(), 0, 1, &vkSet, 0, VK_NULL_HANDLE);

    direction[0U] = 0.0f;
    direction[1U] = 1.0f;
    vkCmdPushConstants(a_cmdBuff, m_ssaoBlurPipeline.getVkPipelineLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(direction), direction);
    vkCmdDispatch(a_cmdBuff, (m_width + 31) / 32, (m_height + 31) / 32, 1);

    etna::set_state(a_cmdBuff, m_ssaoImage.get(), vk::PipelineStageFlagBits2::eFragmentShader, vk::AccessFlagBits2::eShaderRead, vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageAspectFlagBits::eColor);
    etna::flush_barriers(a_cmdBuff);
  }

  //// draw final scene to screen
  //
  {
    auto simpleMaterialInfo = etna::get_shader_program("forward_ssao");

    auto set = etna::create_descriptor_set(simpleMaterialInfo.getDescriptorLayoutId(0), a_cmdBuff,
    {
      etna::Binding {0, constants.genBinding()},
      etna::Binding {1, shadowMap.genBinding(defaultSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)},
      etna::Binding {2, m_ssaoImage.genBinding(defaultSampler.get(), vk::ImageLayout::eShaderReadOnlyOptimal)}
    });

    VkDescriptorSet vkSet = set.getVkSet();

    etna::RenderTargetState renderTargets(a_cmdBuff, {0, 0, m_width, m_height},
      {{.image = a_targetImage, .view = a_targetImageView}},
      {.image = mainViewDepth.get(), .view = mainViewDepth.getView({}), .loadOp = vk::AttachmentLoadOp::eLoad});

    vkCmdBindPipeline(a_cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, m_forwardSSAOPipeline.getVkPipeline());
    vkCmdBindDescriptorSets(a_cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS,
      m_forwardSSAOPipeline.getVkPipelineLayout(), 0, 1, &vkSet, 0, VK_NULL_HANDLE);

    DrawSceneCmd(a_cmdBuff, m_worldViewProj, m_forwardSSAOPipeline.getVkPipelineLayout());
  }

  if (m_input.drawFSQuad) {    
    m_pQuad->RecordCommands(a_cmdBuff, a_targetImage, a_targetImageView, m_ssaoImage, defaultSampler);
  }

  etna::set_state(a_cmdBuff, a_targetImage, vk::PipelineStageFlagBits2::eBottomOfPipe,
    vk::AccessFlags2(), vk::ImageLayout::ePresentSrcKHR,
    vk::ImageAspectFlagBits::eColor);

  etna::finish_frame(a_cmdBuff);

  VK_CHECK_RESULT(vkEndCommandBuffer(a_cmdBuff));
}
