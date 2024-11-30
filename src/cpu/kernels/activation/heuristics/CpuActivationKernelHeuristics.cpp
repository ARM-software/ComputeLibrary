/*
 * Copyright (c) 2017-2024 Arm Limited.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "src/cpu/kernels/activation/heuristics/CpuActivationKernelHeuristics.h"

#include "arm_compute/core/utils/DataTypeUtils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"

#include "src/core/common/Registrars.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/cpu/kernels/activation/list.h"
#include "src/cpu/kernels/logistic/list.h"

#include <map>
#include <vector>

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
namespace heuristics
{
namespace
{

bool is_fp16_lut_supported(ActivationLayerInfo::ActivationFunction func)
{
    return func == ActivationLayerInfo::ActivationFunction::LOGISTIC ||
           func == ActivationLayerInfo::ActivationFunction::TANH;
}

using KernelList = std::vector<CpuActivationKernelHeuristics::ActivationKernel>;
using KernelMap  = std::map<DataType, KernelList>;

static const KernelList fp32_kernels = {
    {"sme2_fp32_logistic",
     [](const ActivationDataTypeISASelectorData &data)
     { return data.f == ActivationLayerInfo::ActivationFunction::LOGISTIC && data.isa.sme2; },
     REGISTER_FP32_SME2(arm_compute::cpu::sme2_fp32_logistic)},
    {"sve_fp32_activation",
     [](const ActivationDataTypeISASelectorData &data)
     { return data.isa.sve && data.f != ActivationLayerInfo::ActivationFunction::GELU; },
     REGISTER_FP32_SVE(arm_compute::cpu::sve_fp32_activation)},
    {"neon_fp32_activation",
     [](const ActivationDataTypeISASelectorData &data)
     {
         ARM_COMPUTE_UNUSED(data);
         return true;
     },
     REGISTER_FP32_NEON(arm_compute::cpu::neon_fp32_activation)},
};

static const KernelList fp16_kernels = {
    {"sve_fp16_activation_lut",
     [](const ActivationDataTypeISASelectorData &data)
     { return data.isa.fp16 && data.isa.sve && is_fp16_lut_supported(data.f); },
     REGISTER_FP16_SVE(arm_compute::cpu::sve_fp16_activation_lut)},
    {"sve_fp16_activation",
     [](const ActivationDataTypeISASelectorData &data)
     { return data.isa.sve && data.isa.fp16 && data.f != ActivationLayerInfo::ActivationFunction::GELU; },
     REGISTER_FP16_SVE(arm_compute::cpu::sve_fp16_activation)},
    {"neon_fp16_activation", [](const ActivationDataTypeISASelectorData &data) { return data.isa.fp16; },
     REGISTER_FP16_NEON(arm_compute::cpu::neon_fp16_activation)},
};

static const KernelList qasymm8_kernels = {
    {"sve2_q8_activation_lut",
     [](const ActivationDataTypeISASelectorData &data) {
         return data.cpumodel == CPUModel::A510 && data.isa.sve2 &&
                data.f != ActivationLayerInfo::ActivationFunction::RELU;
     },
     REGISTER_QASYMM8_SVE2(arm_compute::cpu::sve2_q8_activation_lut)},
#ifdef __aarch64__
    {// Neon LUT implementantion takes precedence
     "neon_q8_activation_lut",
     [](const ActivationDataTypeISASelectorData &data)
     { return data.f != ActivationLayerInfo::ActivationFunction::RELU; },
     REGISTER_Q8_NEON(arm_compute::cpu::neon_q8_activation_lut)},
#endif // __aarch64__
    {"sve2_qu8_activation",
     [](const ActivationDataTypeISASelectorData &data)
     { return data.isa.sve2 && data.f != ActivationLayerInfo::ActivationFunction::GELU; },
     REGISTER_QASYMM8_SVE2(arm_compute::cpu::sve2_qasymm8_activation)},
    {"neon_qu8_activation",
     [](const ActivationDataTypeISASelectorData &data)
     {
         ARM_COMPUTE_UNUSED(data);
         return true;
     },
     REGISTER_QASYMM8_NEON(arm_compute::cpu::neon_qasymm8_activation)},
};

static const KernelList qasymm8_signed_kernels = {
    {"sve2_q8_activation_lut",
     [](const ActivationDataTypeISASelectorData &data) {
         return data.cpumodel == CPUModel::A510 && data.isa.sve2 &&
                data.f != ActivationLayerInfo::ActivationFunction::RELU;
     },
     REGISTER_QASYMM8_SIGNED_SVE2(arm_compute::cpu::sve2_q8_activation_lut)},
#ifdef __aarch64__
    {// Neon LUT implementantion takes precedence
     "neon_q8_activation_lut",
     [](const ActivationDataTypeISASelectorData &data)
     { return data.f != ActivationLayerInfo::ActivationFunction::RELU; },
     REGISTER_Q8_NEON(arm_compute::cpu::neon_q8_activation_lut)},
#endif // __aarch64__
    {"sve2_qs8_activation",
     [](const ActivationDataTypeISASelectorData &data)
     { return data.isa.sve2 && data.f != ActivationLayerInfo::ActivationFunction::GELU; },
     REGISTER_QASYMM8_SIGNED_SVE2(arm_compute::cpu::sve2_qasymm8_signed_activation)},
    {"neon_qs8_activation",
     [](const ActivationDataTypeISASelectorData &data)
     {
         ARM_COMPUTE_UNUSED(data);
         return true;
     },
     REGISTER_QASYMM8_SIGNED_NEON(arm_compute::cpu::neon_qasymm8_signed_activation)},
};

static const KernelList qsymm16_kernels = {
    {"sve2_qs16_activation",
     [](const ActivationDataTypeISASelectorData &data)
     { return data.isa.sve2 && data.f != ActivationLayerInfo::ActivationFunction::GELU; },
     REGISTER_QSYMM16_SVE2(arm_compute::cpu::sve2_qsymm16_activation)},
    {"neon_qs16_activation",
     [](const ActivationDataTypeISASelectorData &data)
     {
         ARM_COMPUTE_UNUSED(data);
         return true;
     },
     REGISTER_QSYMM16_NEON(arm_compute::cpu::neon_qsymm16_activation)},
};

static const KernelMap kernels = {{DataType::F32, fp32_kernels},
                                  {DataType::F16, fp16_kernels},
                                  {DataType::QASYMM8, qasymm8_kernels},
                                  {DataType::QASYMM8_SIGNED, qasymm8_signed_kernels},
                                  {DataType::QSYMM16, qsymm16_kernels}};

/** Find the index of the first element greater than the input
 * @note binary search does not provide much value over the small array,
 *       therefore we keep the implementation simple.
 *
 * @param arr input array
 * @param len length of the input array
 * @param x element to compare
 * @return the index found
 */
size_t find_ind_lte_elm(const size_t *arr, size_t len, size_t x)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(arr);
    for (size_t i = 0; i < len; ++i)
    {
        if (x <= arr[i])
        {
            return i;
        }
    }

    return len - 1;
}

size_t calculate_mws(const CPUModel cpu_model, DataType dtype, const ActivationLayerInfo &act_info, size_t problem_size)
{
    // This number is loosely chosen as threading overhead in each platform varies wildly.
    size_t mws = 1529;

    if (cpu_model == CPUModel::V1)
    {
        // If max_threads is smaller than the number of threads suggested in the heuristics,
        //
        const size_t max_threads = NEScheduler::get().num_threads();

        constexpr int32_t   compute_heavy_arr_fp32_len                            = 26;
        static const size_t compute_heavy_arr_fp32[2][compute_heavy_arr_fp32_len] = {
            {2000,  4000,  5000,   6000,   8000,   9000,   10000,  20000,  30000,  40000,  50000,  60000,   70000,
             80000, 90000, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000, 2000000},
            {1, 2, 3, 4, 5, 6, 7, 9, 12, 14, 15, 18, 20, 22, 25, 29, 36, 43, 48, 53, 57, 58, 59, 60, 62, max_threads}};

        constexpr int32_t   compute_light_arr_fp32_len                            = 20;
        static const size_t compute_light_arr_fp32[2][compute_light_arr_fp32_len] = {
            {30000,  40000,  50000,  70000,  80000,   90000,   100000,  200000,  300000,  400000,
             500000, 600000, 700000, 900000, 1000000, 2000000, 3000000, 4000000, 5000000, 6000000},
            {1, 2, 3, 4, 6, 8, 10, 13, 15, 18, 21, 23, 24, 25, 30, 38, 45, 53, 60, max_threads}};

        constexpr int32_t   compute_heavy_arr_fp16_len                            = 24;
        static const size_t compute_heavy_arr_fp16[2][compute_heavy_arr_fp16_len] = {
            {10000,  30000,  40000,  50000,   60000,   70000,   80000,   90000,   100000,  200000,  300000,   400000,
             500000, 800000, 900000, 1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 8000000, 10000000, 20000000},
            {1, 2, 3, 5, 6, 7, 8, 10, 13, 17, 20, 23, 25, 28, 32, 37, 43, 49, 55, 58, 60, 61, 62, max_threads}};

        constexpr int32_t   compute_light_arr_fp16_len                            = 20;
        static const size_t compute_light_arr_fp16[2][compute_light_arr_fp16_len] = {
            {30000,  40000,  50000,  70000,  80000,   90000,   100000,  200000,  300000,  400000,
             500000, 600000, 700000, 900000, 1000000, 2000000, 3000000, 4000000, 5000000, 6000000},
            {1, 2, 3, 4, 6, 8, 10, 13, 15, 18, 21, 23, 24, 25, 30, 38, 45, 53, 60, max_threads}};

        constexpr int32_t   s8_arr_len            = 24;
        static const size_t s8_arr[2][s8_arr_len] = {
            {7000,   8000,   9000,   10000,  20000,  30000,  40000,  60000,   70000,   90000,   100000,  200000,
             300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000, 2000000, 3000000, 8000000, 9000000},
            {1, 2, 3, 4, 6, 7, 10, 11, 13, 15, 19, 23, 26, 31, 37, 40, 44, 48, 52, 54, 58, 61, 62, max_threads}};

        const size_t dtype_len = data_size_from_type(dtype);

        const size_t *size_arr    = nullptr;
        const size_t *nthread_arr = nullptr;
        size_t        arr_len     = 0;

        switch (act_info.activation())
        {
            case ActivationLayerInfo::ActivationFunction::LOGISTIC:
            case ActivationLayerInfo::ActivationFunction::SWISH:
            case ActivationLayerInfo::ActivationFunction::ELU:
            case ActivationLayerInfo::ActivationFunction::GELU:
            case ActivationLayerInfo::ActivationFunction::SOFT_RELU:
            case ActivationLayerInfo::ActivationFunction::TANH:
            {
                switch (dtype_len)
                {
                    case 4:
                        size_arr    = &compute_heavy_arr_fp32[0][0];
                        nthread_arr = &compute_heavy_arr_fp32[1][0];
                        arr_len     = compute_heavy_arr_fp32_len;
                        break;
                    case 2:
                        size_arr    = &compute_heavy_arr_fp16[0][0];
                        nthread_arr = &compute_heavy_arr_fp16[1][0];
                        arr_len     = compute_heavy_arr_fp16_len;
                        break;
                    case 1:
                    default:
                        size_arr    = &s8_arr[0][0];
                        nthread_arr = &s8_arr[1][0];
                        arr_len     = s8_arr_len;
                        break;
                }
                break;
            }
            default:
            {
                switch (dtype_len)
                {
                    case 4:
                        size_arr    = &compute_light_arr_fp32[0][0];
                        nthread_arr = &compute_light_arr_fp32[1][0];
                        arr_len     = compute_light_arr_fp32_len;
                        break;
                    case 2:
                        size_arr    = &compute_light_arr_fp16[0][0];
                        nthread_arr = &compute_light_arr_fp16[1][0];
                        arr_len     = compute_light_arr_fp16_len;
                        break;
                    case 1:
                    default:
                        size_arr    = &s8_arr[0][0];
                        nthread_arr = &s8_arr[1][0];
                        arr_len     = s8_arr_len;
                        break;
                }
                break;
            }
        }

        const size_t ind      = find_ind_lte_elm(size_arr, arr_len, problem_size);
        const size_t nthreads = std::min(nthread_arr[ind], max_threads);
        mws                   = (problem_size + nthreads - 1) / nthreads;
    }

    return mws;
}

} // namespace

void CpuActivationKernelHeuristics::choose_kernel(ActivationDataTypeISASelectorData &selector)
{
    const auto &klist = kernels.find(selector.dt);
    if (klist == kernels.end())
    {
        return;
    }

    for (const auto &uk : klist->second)
    {
        if (uk.is_selected(selector) && uk.ukernel != nullptr)
        {
            _kernel = &uk;
            return;
        }
    }
}

CpuActivationKernelHeuristics::CpuActivationKernelHeuristics(const ITensorInfo         *src,
                                                             const ITensorInfo         *dst,
                                                             const ActivationLayerInfo &activation_info)
{
    ARM_COMPUTE_UNUSED(dst);

    // Set kernel
    const DataType                    dtype = src->data_type();
    ActivationDataTypeISASelectorData selector{dtype, CPUInfo::get().get_cpu_model(), CPUInfo::get().get_isa(),
                                               activation_info.activation()};
    const CPUModel                    cpu_model = CPUInfo::get().get_cpu_model();
    choose_kernel(selector);

    // Set window and scheduling hint
    int split_dim;
    std::tie(_window, split_dim) = calculate_squashed_or_max_window(*src);

    // Collapse window with SME kernels in Y-Dim
    if (std::string(_kernel->name) == "sme2_fp32_logistic")
    {
        _window = _window.collapse(_window, Window::DimY);
    }

    _hint = IScheduler::Hints(split_dim);

    // Set minimum workload size
    if (split_dim == Window::DimX)
    {
        _mws = calculate_mws(cpu_model, src->data_type(), activation_info.activation(), src->tensor_shape().x());
    }
}

/** Return minimum workload size
 *
 * @return Minimum workload size for requested configuration.
 */
size_t CpuActivationKernelHeuristics::mws() const
{
    return _mws;
}

/** Return kernel's execution window
 *
 * @return The execution window
 */
const Window &CpuActivationKernelHeuristics::window() const
{
    return _window;
}

/** Return the kernel to run
 *
 * @return The function pointer to the chosen kernel
 */
const CpuActivationKernelHeuristics::ActivationKernel *CpuActivationKernelHeuristics::kernel()
{
    return _kernel;
}

/** Return the scheduling hint e.g. dimension(s) to split
 *
 * @return an instance of @ref IScheduler::Hints to describe the scheduling hints
 */
const IScheduler::Hints &CpuActivationKernelHeuristics::scheduler_hint() const
{
    return _hint;
}
} // namespace heuristics
} // namespace kernels
} // namespace cpu
} // namespace arm_compute
