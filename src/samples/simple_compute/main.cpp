#include "simple_compute.h"

#include <chrono>
#include <cstdint>
#include <random>

template<typename Func>
void RunWithTimer(const char* timeLabel, Func func) {
  using namespace std::chrono;

  auto start = high_resolution_clock::now();
  func();
  auto end = high_resolution_clock::now();

  auto time = duration_cast<nanoseconds>(end - start).count() * 1e-6f;
  std::cout << "Timer [" << timeLabel << "]: " << time << " ms\n";
}

void TestCPU(const std::vector<float>& array)
{
  float sum = 0.0f;

  RunWithTimer("CPU", [&]() {
    std::int32_t n = static_cast<std::int32_t>(array.size());

    std::vector<float> result(n);
    for (std::int32_t elemIdx = 0; elemIdx < n; ++elemIdx)
    {
      float curResult = 0.0;

      for (std::int32_t windowOffset = -3; windowOffset <= 3; ++windowOffset)
      {
        std::int32_t idx = elemIdx + windowOffset;

        if (idx >= 0 && idx < n)
        {
          curResult += array[idx];
        }
      }

      curResult /= 7.0f;

      result[elemIdx] = array[elemIdx] - curResult;
      sum += result[elemIdx];
    }
  });

  std::cout << "Sum [CPU]: " << sum << std::endl;
}

void TestCompute(SimpleCompute& simple_compute, const std::vector<float>& array)
{
  float sum = 0.f;

  RunWithTimer("Compute:TOTAL", [&]() {
    RunWithTimer("Compute:UpdateBuffer", [&]() {
      simple_compute.GetCopyEngine()->UpdateBuffer(simple_compute.GetInputBuffer(), 0, array.data(), sizeof(float) * array.size());
    });

    RunWithTimer("Compute:Dispatch", [&]() {
      simple_compute.Execute();
    });

    std::vector<float> result(array.size());
    RunWithTimer("Compute:ReadBuffer", [&]() {
      simple_compute.GetCopyEngine()->ReadBuffer(simple_compute.GetOutputBuffer(), 0, result.data(), sizeof(float) * result.size());
    });

    RunWithTimer("Compute:CalcSum", [&]() {
      for (auto x : result)
      {
        sum += x;
      }
    });
  });

  std::cout << "Sum [Compute]: " << sum << std::endl;
}

void TestComputeSharedMemory(SimpleCompute& simple_compute, const std::vector<float>& array)
{
  float sum = 0.f;

  RunWithTimer("ComputeSharedMemory:TOTAL", [&]() {
    RunWithTimer("ComputeSharedMemory:UpdateBuffer", [&]() {
      simple_compute.GetCopyEngine()->UpdateBuffer(simple_compute.GetInputBuffer(), 0, array.data(), sizeof(float) * array.size());
    });

    RunWithTimer("ComputeSharedMemory:Dispatch", [&]() {
      simple_compute.Execute();
    });

    // std::vector<float> result(array.size());
    std::vector<float> result(simple_compute.GetWorkGroupCount());
    RunWithTimer("ComputeSharedMemory:ReadBuffer", [&]() {
      simple_compute.GetCopyEngine()->ReadBuffer(simple_compute.GetOutputBuffer(), 0, result.data(), sizeof(float) * result.size());
    });

    RunWithTimer("ComputeSharedMemory:CalcSum", [&]() {
      for (auto x : result) {
        sum += x;
      }
    });
  });

  std::cout << "Sum [ComputeSharedMemory]: " << sum << std::endl;
}

std::vector<float> CreateRandomArray(std::size_t size, float rangeFrom, float rangeTo) {
  static std::random_device s_device;
  static std::mt19937 s_generator{s_device()};

  std::uniform_real_distribution<float> dist(rangeFrom, rangeTo);

  std::vector<float> array(size);
  for (auto& x : array) {
    x = dist(s_generator);
  }

  return array;
}

int main()
{
  constexpr int LENGTH = 1'000'000;
  constexpr int VULKAN_DEVICE_ID = 0;

  auto simple_compute = std::make_unique<SimpleCompute>(LENGTH);
  if(simple_compute == nullptr)
  {
    std::cout << "Can't create render of specified type" << std::endl;
    return 1;
  }

  simple_compute->InitVulkan(nullptr, 0, VULKAN_DEVICE_ID);

  auto array = CreateRandomArray(LENGTH, -1000.0f, 1000.0f);

  TestCPU(array);
  simple_compute->BuildPipeline("../resources/shaders/simple.comp.spv");
  TestCompute(*simple_compute, array);
  simple_compute->BuildPipeline("../resources/shaders/simple_shared_memory.comp.spv");
  TestComputeSharedMemory(*simple_compute, array);

  return 0;
}
