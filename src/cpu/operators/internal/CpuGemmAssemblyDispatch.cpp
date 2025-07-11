/*
 * Copyright (c) 2018-2025 Arm Limited.
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
#include "src/cpu/operators/internal/CpuGemmAssemblyDispatch.h"

#include "arm_compute/core/Error.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"

#include "src/core/CPP/Validate.h"
#include "src/core/helpers/MemoryHelpers.h"
#include "src/core/utils/AssemblyUtils.h"
#include "src/cpu/kernels/assembly/arm_gemm.hpp"
#include "src/cpu/kernels/assembly/CpuGemmAssemblyWrapperKernel.h"
#include "src/cpu/operators/CpuTranspose.h"
#include "src/cpu/utils/CpuAuxTensorHandler.h"

#include <arm_neon.h>

namespace arm_compute
{
namespace cpu
{
namespace
{
/** Run pretranspose_B_array in parallel (1D static scheduling)
 *
 * @tparam TypeInput
 * @tparam TypeWeight
 * @tparam TypeOutput
 *
 * @param[in] gemm_asm         GemmCommon kernel to run
 * @param[in] dst              Pretransposed B array
 * @param[in] src              B array to be pretransposed
 * @param[in] src_ld           Stride in y
 * @param[in] src_multi_stride Stride in z ("multi")
 * @param[in] num_threads      Number of threads to run this method. Must be >= 1
 */
template <typename TypeInput, typename TypeWeight, typename TypeOutput>
void run_parallel_pretranspose_B_array(arm_gemm::GemmCommon<TypeInput, TypeWeight, TypeOutput> *gemm_asm,
                                       ITensor                                                 *dst,
                                       const TypeWeight                                        *src,
                                       int                                                      src_ld,
                                       int                                                      src_multi_stride,
                                       unsigned int                                             num_threads,
                                       bool                                                     transpose)
{
    ARM_COMPUTE_ERROR_ON(gemm_asm == nullptr);
    ARM_COMPUTE_ERROR_ON(num_threads == 0);
    // The window size is also the total workload size
    const unsigned int wsize = gemm_asm->get_B_pretranspose_window_size();

    const int workload_size       = std::min(wsize, num_threads);
    const int chunks_per_workload = std::floor(wsize / workload_size);

    std::vector<IScheduler::Workload> workloads(workload_size);
    for (int t = 0; t < workload_size; ++t)
    {
        workloads[t] = [=](const ThreadInfo &info)
        {
            const unsigned int start = ((info.thread_id) * chunks_per_workload);
            unsigned int       end;
            if (info.thread_id < workload_size - 1)
            {
                end = ((info.thread_id + 1) * chunks_per_workload);
            }
            else
            {
                end = wsize;
            }
            ARM_COMPUTE_ERROR_ON(start > end);
            if (start < end)
            {
                gemm_asm->pretranspose_B_array_part(dst->buffer(), src, src_ld, src_multi_stride, transpose, start,
                                                    end);
            }
        };
    }
    NEScheduler::get().run_tagged_workloads(workloads, "CpuGemmAssemblyDispatch/pretranspose_B_array");
}
} // namespace

using namespace arm_compute::experimental;

namespace
{
struct Params
{
    unsigned int M;
    unsigned int N;
    unsigned int K;
    unsigned int batches;
    unsigned int multis;
    unsigned int sections;
    bool         indirect;
};

Params extract_parameters(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *d, const AsmGemmInfo &info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(a, b, d);
    Params p{/* M */ static_cast<unsigned int>(d->tensor_shape().y()),
             /* N */ static_cast<unsigned int>(d->tensor_shape().x()),
             /* K */ static_cast<unsigned int>(a->tensor_shape().x()),
             /* batches */ 1,
             /* multis */ 1,
             /* sections */ 1,
             /* indirect */ false};

    if (info.method == AsmConvMethod::Conv || info.method == AsmConvMethod::Indirect)
    {
        p.indirect = true;
        p.sections = b->tensor_shape()[2] * b->tensor_shape()[3];
    }
    else
    {
        p.multis  = b->tensor_shape().z();
        p.batches = d->tensor_shape().total_size_upper(2) / p.multis;
    }

    // Update M in case of GEMM3D for output
    if (info.depth_output_gemm3d != 0)
    {
        p.M       = d->tensor_shape().y() * d->tensor_shape().z();
        p.batches = d->tensor_shape().total_size_upper(3) / p.multis;
    }

    return p;
}

/** Fallback in case ACL doesn't have a function */
template <typename TypeInput, typename TypeWeight, typename TypeOutput, class OutputStage = arm_gemm::Nothing>
class Fallback : public CpuGemmAssemblyDispatch::IFallback
{
public:
    /** Initialise the functions's input and output.
     *
     * @param[in]  a         Input tensor containing the Matrix A.
     * @param[in]  b         Input tensor containing the Matrix B.
     * @param[in]  c         Input tensor containing the Matrix C.
     * @param[out] d         Output tensor to store the result of matrix multiplication.
     * @param[in]  args      Matrix multiplication information.
     * @param[in]  gemm_info GEMM meta-data
     * @param[in]  os        Output stage meta-data.
     */
    void configure(const ITensorInfo *a,
                   const ITensorInfo *b,
                   const ITensorInfo *c,
                   ITensorInfo       *d,
                   arm_gemm::GemmArgs args,
                   const AsmGemmInfo &gemm_info,
                   const OutputStage &os = {});

    /** Set requantization shifts to be used
     *
     * @param[in] shifts Requantization shifts
     *
     * @return Pointer to the shift data
     */
    /** Set requantization data to be used
      *
      *
      * @param shifts       Requantization shifts
      * @param multipliers  Requantization multipliers
      *
      * @return A tuple with the pointers to the shift and multiplier data respectively
      */
    std::tuple<bool, const int32_t *, const int32_t *, const int32_t *>
    set_requantize_data(const std::vector<int32_t> &shifts, const std::vector<int32_t> &multipliers);

    // Inherited methods overridden:
    void                             run(ITensorPack &tensors) override;
    void                             prepare(ITensorPack &tensors) override;
    bool                             is_configured() const override;
    experimental::MemoryRequirements workspace() const override;
    bool                             isVarWeightsKernel() const override
    {
        if (!_gemm_kernel_asm)
        {
            return false;
        }
        const arm_compute::WeightFormat wf =
            assembly_utils::map_to_arm_compute_weight_format(_gemm_kernel_asm->get_config().weight_format);
        return wf != arm_compute::WeightFormat::UNSPECIFIED && wf != arm_compute::WeightFormat::ANY;
    }

    void update_quantization_parameters(const GEMMLowpOutputStageInfo &output_info,
                                        const QuantizationInfo        &a,
                                        const QuantizationInfo        &b,
                                        const bool                     is_prepared,
                                        const bool                     negated_offsets) override
    {
        const int32_t negation = negated_offsets ? 1 : -1;
        const int32_t a_offset = -a.uniform().offset * negation;
        const int32_t b_offset = -b.uniform().offset * negation;

        arm_gemm::Requantize32 gemm_requant_info{};
        if (output_info.gemmlowp_shifts.size() > 1)
        {
            const auto requantize_data =
                this->set_requantize_data(output_info.gemmlowp_multipliers, output_info.gemmlowp_shifts);
            gemm_requant_info = arm_gemm::Requantize32(
                nullptr, 0, a_offset, b_offset, output_info.gemmlowp_offset,
                (std::get<0>(requantize_data)) ? std::get<1>(requantize_data) : nullptr, std::get<2>(requantize_data),
                std::get<3>(requantize_data), output_info.gemmlowp_min_bound, output_info.gemmlowp_max_bound);
        }
        else
        {
            gemm_requant_info = arm_gemm::Requantize32(nullptr, 0, a_offset, b_offset, output_info.gemmlowp_offset,
                                                       -output_info.gemmlowp_shift, output_info.gemmlowp_multiplier,
                                                       output_info.gemmlowp_min_bound, output_info.gemmlowp_max_bound);
        }

        _gemm_kernel_asm->update_quantization_parameters(gemm_requant_info);

        // After update_quantization_parameters(), window may change, reconfigure it.
        auto *opt = reinterpret_cast<kernel::CpuGemmAssemblyWrapperKernel<TypeInput, TypeWeight, TypeOutput> *>(
            _optimised_kernel.get());
        const Window win = to_window(_gemm_kernel_asm->get_window_size());
        opt->configure_window(win);

        _is_prepared = is_prepared;
    }

private:
    enum AuxTensorIdx
    {
        AsmGemmWorkspace = 0,
        PrePretransposedB, /* Transposed B (rhs) before being passed to gemm or pretranspose_B_array */
        Pretranspose,
        Count
    };

    /** Configure the indirect buffer
     *
     * @param[in]  a    Input tensor containing the Matrix A.
     * @param[in]  b    Input tensor containing the Matrix B.
     * @param[out] d    Output tensor to store the result of matrix multiplication.
     * @param[in]  info GEMM meta-data
     */
    void configure_indirect(const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *d, const AsmGemmInfo &info);
    /** Prepare the indirect buffer */
    void prepare_indirect_buffer(ITensorPack &tensors);

    /** Operator to transpose B before gemm or pretranspose_B_array*/
    std::unique_ptr<CpuTranspose> _pre_pretranspose_b{nullptr};
    /** Assembly Gemm kernel */
    std::shared_ptr<arm_gemm::GemmCommon<TypeInput, TypeWeight, TypeOutput>> _gemm_kernel_asm{nullptr};
    /** Optimised Arm® Neon™ kernel */
    std::unique_ptr<INEKernel> _optimised_kernel{nullptr};
    /** Assembly GEMM workspace tensor info */
    TensorInfo _workspace_info{};
    /** Pre-pre-transposed B tensor info */
    TensorInfo _pre_pretransposed_b_info{};
    /** Pre-transpose tensor info */
    TensorInfo _pretranspose_info{};
    /** Prepared flag */
    bool _is_prepared{false};
    /** GEMM meta-data */
    AsmGemmInfo _gemm_info{};
    /** Per channel quantization shifts */
    std::vector<int32_t> _shifts{};
    std::vector<int32_t> right_shifts{};
    std::vector<int32_t> left_shifts{};
    /** Per channel quantization multipliers */
    std::vector<int32_t> _multipliers{};
    /** Indirect buffer */
    std::vector<const TypeInput *const *> _indirect_arg{};
    std::vector<const TypeInput *>        _indirect_buf{};
    std::vector<TypeInput>                _indirect_pad{};
    arm_gemm::ConvolutionParameters       _cp{};
    experimental::MemoryRequirements      _aux_mem{Count};
    bool                                  _B_pretranspose_required{false};
    bool                                  _is_b_constant{true};
    bool                                  _is_c_constant{true};
    bool                                  _run_pre_pretranspose_b{false};
    bool                                  _B_pre_pretranspose_required{false};
};

template <typename TypeInput, typename TypeWeight, typename TypeOutput, class OutputStage>
std::tuple<bool, const int32_t *, const int32_t *, const int32_t *>
Fallback<TypeInput, TypeWeight, TypeOutput, OutputStage>::set_requantize_data(const std::vector<int32_t> &shifts,
                                                                              const std::vector<int32_t> &multipliers)
{
    _multipliers   = multipliers;
    _shifts        = shifts;
    bool need_left = false;
    for (const auto s : _shifts)
    {
        left_shifts.push_back(std::max(-s, int32_t(0)));
        right_shifts.push_back(std::min(-s, int32_t(0)));
        if (s < 0 && !need_left)
        {
            need_left = true;
        }
    }
    return std::make_tuple(need_left, left_shifts.data(), right_shifts.data(), _multipliers.data());
}

template <typename TypeInput, typename TypeWeight, typename TypeOutput, class OutputStage>
void Fallback<TypeInput, TypeWeight, TypeOutput, OutputStage>::prepare_indirect_buffer(ITensorPack &tensors)
{
    auto             a              = tensors.get_const_tensor(TensorType::ACL_SRC_0);
    const TypeInput *A_ptr          = reinterpret_cast<TypeInput *>(a->buffer());
    const int        multis         = 1;
    const int        batches        = a->info()->tensor_shape().total_size_upper(3);
    const size_t     stride_A       = a->info()->strides_in_bytes().y() / sizeof(TypeInput);
    const size_t     batch_stride_A = a->info()->strides_in_bytes()[3] / sizeof(TypeInput);
    const size_t     multi_stride_A = a->info()->strides_in_bytes()[4] / sizeof(TypeInput);

    const size_t output_hw    = _cp.output_height * _cp.output_width;
    const int    batch_size   = _cp.kernel_height * _cp.kernel_width * output_hw * sizeof(TypeInput);
    const size_t batch_stride = batch_size / sizeof(TypeInput);
    const int    multi_size   = batch_size * batches;
    const size_t multi_stride = multi_size / sizeof(TypeInput);

    for (int64_t m = 0; m < multis; m++)
    {
        for (int64_t b = 0; b < batches; b++)
        {
            for (int64_t output_y = 0; output_y < _cp.output_height; output_y++)
            {
                for (int64_t output_x = 0; output_x < _cp.output_width; output_x++)
                {
                    int64_t output_xy = (output_y * _cp.output_width) + output_x;

                    for (int64_t kernel_y = 0; kernel_y < _cp.kernel_height; kernel_y++)
                    {
                        for (int64_t kernel_x = 0; kernel_x < _cp.kernel_width; kernel_x++)
                        {
                            int64_t input_x   = (output_x * _cp.output_stride_w) + kernel_x - _cp.padding_left;
                            int64_t input_y   = (output_y * _cp.output_stride_h) + kernel_y - _cp.padding_top;
                            int64_t kernel_xy = (kernel_y * _cp.kernel_width) + kernel_x;
                            int64_t input_xy  = (input_y * _cp.input_width) + input_x;

                            if (input_x < 0 || input_x >= _cp.input_width || input_y < 0 || input_y >= _cp.input_height)
                            {
                                _indirect_buf[m * multi_stride + b * batch_stride + kernel_xy * output_hw + output_xy] =
                                    _indirect_pad.data();
                            }
                            else
                            {
                                _indirect_buf[m * multi_stride + b * batch_stride + kernel_xy * output_hw + output_xy] =
                                    A_ptr + (m * multi_stride_A + b * batch_stride_A + input_xy * stride_A);
                            }
                        }
                    }
                }
            }
        }
    }
}

template <typename TypeInput, typename TypeWeight, typename TypeOutput, class OutputStage>
void Fallback<TypeInput, TypeWeight, TypeOutput, OutputStage>::configure_indirect(const ITensorInfo *a,
                                                                                  const ITensorInfo *b,
                                                                                  const ITensorInfo *d,
                                                                                  const AsmGemmInfo &info)
{
    ARM_COMPUTE_ERROR_ON(!(info.method == AsmConvMethod::Conv || info.method == AsmConvMethod::Indirect));

    float zeropad = 0.f;
    if (is_data_type_quantized(a->data_type()))
    {
        zeropad = a->quantization_info().uniform().offset;
    }

    const auto input_width    = static_cast<int64_t>(a->tensor_shape()[1]);
    const auto input_height   = static_cast<int64_t>(a->tensor_shape()[2]);
    const auto input_channels = static_cast<int64_t>(a->tensor_shape()[0]);
    const auto kernel_width   = static_cast<int64_t>(b->tensor_shape()[2]);
    const auto kernel_height  = static_cast<int64_t>(b->tensor_shape()[3]);
    const auto output_width   = static_cast<int64_t>(d->tensor_shape()[1]);
    const auto output_height  = static_cast<int64_t>(d->tensor_shape()[2]);

    _cp = {input_width,
           input_height,
           input_channels,
           kernel_width,
           kernel_height,
           output_width,
           output_height,
           info.ps_info.stride().first,
           info.ps_info.stride().second,
           1,
           1,
           info.padding_top,
           info.padding_left,
           zeropad};

    if (info.method == AsmConvMethod::Conv)
    {
        _gemm_kernel_asm->set_convolution_parameters(_cp);
    }

    if (info.method == AsmConvMethod::Indirect)
    {
        const unsigned int multis    = 1;
        const unsigned int batches   = a->tensor_shape().total_size_upper(3);
        const unsigned int kernel_hw = _cp.kernel_width * _cp.kernel_height;
        const unsigned int output_hw = _cp.output_width * _cp.output_height;

        using TypeInputPtr        = TypeInput *;
        const int    batch_size   = kernel_hw * output_hw * sizeof(TypeInputPtr);
        const size_t batch_stride = batch_size / sizeof(TypeInputPtr);
        const int    multi_size   = batch_size * batches;
        const size_t multi_stride = multi_size / sizeof(TypeInputPtr);

        _indirect_buf = std::vector<const TypeInput *>(multi_size * multis);
        _indirect_arg = std::vector<const TypeInput *const *>(sizeof(TypeInput **) * kernel_hw * multis * batches);
        _indirect_pad = std::vector<TypeInput>(_cp.input_channels, TypeInput(zeropad));

        // Set indirect argument
        int64_t pos = 0;
        for (int64_t m = 0; m < multis; m++)
        {
            for (int64_t b = 0; b < batches; b++)
            {
                for (int64_t kernel_xy = 0; kernel_xy < kernel_hw; kernel_xy++)
                {
                    _indirect_arg[pos++] = &_indirect_buf[m * multi_stride + b * batch_stride + kernel_xy * output_hw];
                }
            }
        }

        _gemm_kernel_asm->set_indirect_parameters(a->tensor_shape()[0], _indirect_arg.data());
    }
}

template <typename TypeInput, typename TypeWeight, typename TypeOutput, class OutputStage>
void Fallback<TypeInput, TypeWeight, TypeOutput, OutputStage>::configure(const ITensorInfo *a,
                                                                         const ITensorInfo *b,
                                                                         const ITensorInfo *c,
                                                                         ITensorInfo       *d,
                                                                         arm_gemm::GemmArgs args,
                                                                         const AsmGemmInfo &gemm_info,
                                                                         const OutputStage &os)
{
    _is_b_constant = b->are_values_constant();
    _is_c_constant = c ? c->are_values_constant() : true;

    _gemm_kernel_asm = arm_gemm::gemm<TypeInput, TypeWeight, TypeOutput, OutputStage>(args, os);
    if (_gemm_kernel_asm == nullptr)
    {
        //configuration not supported: Leave function unconfigured:
        return;
    }

    arm_gemm::GemmConfig gemm_cfg = _gemm_kernel_asm->get_config();

    // arm_compute wrapper for the Gemm object (see above)
    auto acl_gemm_wrapper = std::make_unique<kernel::CpuGemmAssemblyWrapperKernel<TypeInput, TypeWeight, TypeOutput>>();
    ARM_COMPUTE_ERROR_ON(acl_gemm_wrapper == nullptr);
    acl_gemm_wrapper->configure(_gemm_kernel_asm.get(), gemm_cfg.filter);
    const size_t       workspace_size = _gemm_kernel_asm->get_working_size();
    const unsigned int alignment      = 4096;
    _workspace_info                   = TensorInfo(TensorShape(workspace_size), 1, DataType::U8);
    _aux_mem[AsmGemmWorkspace] =
        MemoryInfo(offset_int_vec(AsmGemmWorkspace), MemoryLifetime::Temporary, workspace_size, alignment);

    //if we disable this code below in brackets then ConvLayer deadlocks when threads > 1 and
    //the shapes are In=1x1x1024 Weights=1x1x1024x1001 Biases=1001 Out=1x1x1001
    {
        const unsigned int window_size = _gemm_kernel_asm->get_window_size().total_size();
        if (window_size < static_cast<unsigned int>(args._maxthreads))
        {
            _gemm_kernel_asm->set_nthreads(window_size);
        }
    }

    _optimised_kernel = std::move(acl_gemm_wrapper);
    _gemm_info        = gemm_info;

    // Check if we need to pre-pretranspose B. Fixed format kernels need no pre-pretranspose.
    _B_pre_pretranspose_required = _gemm_info.transpose_b && !isVarWeightsKernel();
    _B_pretranspose_required     = _gemm_kernel_asm->B_pretranspose_required();

    const bool kernel_supports_transpose = _gemm_kernel_asm->B_pretranspose_supports_transpose();
    const bool kernel_can_fuse_transpose = _B_pretranspose_required && kernel_supports_transpose;
    _run_pre_pretranspose_b              = _B_pre_pretranspose_required && !kernel_can_fuse_transpose;

    if (_run_pre_pretranspose_b)
    {
        _pre_pretranspose_b = std::make_unique<CpuTranspose>();
        _pre_pretranspose_b->configure(b, &_pre_pretransposed_b_info);
        MemoryLifetime lifetime;
        if (_is_b_constant)
        {
            if (_B_pretranspose_required)
            {
                // PrePretransposedB tensor is only used in prepare(), but is then succeeded by Pretranspose
                // So PrePretransposedB can be freed inside prepare()
                lifetime = MemoryLifetime::Prepare;
            }
            else
            {
                // PrePretransposedB tensor is only used in prepare(), but is the final transformation of B
                // So PrePretransposedB needs to persist beyond prepare()
                lifetime = MemoryLifetime::Persistent;
            }
        }
        else
        {
            // PrePretransposedB tensor is always used in run() and doesn't need to persist
            lifetime = MemoryLifetime::Temporary;
        }
        // Forcing 128-byte alignment (required by 32-bit kernels)
        const unsigned int alignment = 128;
        _aux_mem[PrePretransposedB] =
            MemoryInfo(offset_int_vec(PrePretransposedB), lifetime, _pre_pretransposed_b_info.total_size(), alignment);
    }

    // Check for pre-transposed support
    if (_B_pretranspose_required)
    {
        // Fixed format kernels need no pretranspose.
        ARM_COMPUTE_ERROR_ON(arm_compute::is_fixed_format(
            assembly_utils::map_to_arm_compute_weight_format(_gemm_kernel_asm->get_config().weight_format)));
        // Forcing 128-byte alignment (required by 32-bit kernels)
        const unsigned int alignment           = 128;
        const size_t       B_pretranspose_size = _gemm_kernel_asm->get_B_pretransposed_array_size();
        _pretranspose_info                     = TensorInfo(TensorShape(B_pretranspose_size), 1, DataType::U8);
        MemoryLifetime lifetime = _is_b_constant ? MemoryLifetime::Persistent : MemoryLifetime::Temporary;
        _aux_mem[Pretranspose]  = MemoryInfo(offset_int_vec(Pretranspose), lifetime, B_pretranspose_size, alignment);
    }

    // Handle indirect GEMM convolution
    if (gemm_info.method == AsmConvMethod::Conv || gemm_info.method == AsmConvMethod::Indirect)
    {
        configure_indirect(a, b, d, gemm_info);
    }

    if (std::is_same<OutputStage, arm_gemm::DequantizeFloat>::value)
    {
        // Output dequantization is just the two src scales multiplied together
        _gemm_kernel_asm->set_dequantize_scale(a->quantization_info().uniform().scale *
                                               b->quantization_info().uniform().scale);
    }
}

template <typename TypeInput, typename TypeWeight, typename TypeOutput, class OutputStage>
void Fallback<TypeInput, TypeWeight, TypeOutput, OutputStage>::prepare(ITensorPack &tensors)
{
    if (!_is_prepared)
    {
        auto b = tensors.get_const_tensor(TensorType::ACL_SRC_1);
        auto c = tensors.get_const_tensor(TensorType::ACL_SRC_2);
        ARM_COMPUTE_ERROR_ON_NULLPTR(b);

        // Setup up matrix bias in the assembly kernel, it's just a pointer to matrix C.
        if (c && c->info()->data_type() == DataType::S32)
        {
            _gemm_kernel_asm->set_quantized_bias(
                reinterpret_cast<const int32_t *>(c->buffer() + c->info()->offset_first_element_in_bytes()), 0);
        }
        const ITensor *b_to_use = b;

        // Pre-pretranspose B if required
        CpuAuxTensorHandler pre_pretransposed_b(
            offset_int_vec(PrePretransposedB), _pre_pretransposed_b_info, tensors,
            /*pack_inject: no need to inject into tensors*/
            false,
            /*bypass_alloc: no need to allocate if pre-pretranspose B is not required as this handle will not be used*/
            !_run_pre_pretranspose_b);

        if (_run_pre_pretranspose_b)
        {
            ARM_COMPUTE_ERROR_ON(_pre_pretranspose_b == nullptr);
            ITensorPack pre_pretranspose_pack{{ACL_SRC, b_to_use}, {ACL_DST, pre_pretransposed_b.get()}};
            _pre_pretranspose_b->run(pre_pretranspose_pack);
            b_to_use = pre_pretransposed_b.get();
        }

        // Pretranspose B if required
        if (_B_pretranspose_required)
        {
            // Fixed format kernels need no pretranspose.
            ARM_COMPUTE_ERROR_ON(arm_compute::is_fixed_format(
                assembly_utils::map_to_arm_compute_weight_format(_gemm_kernel_asm->get_config().weight_format)));
            const int  ldb     = b_to_use->info()->strides_in_bytes().y() / b_to_use->info()->element_size();
            const auto in1_ptr = reinterpret_cast<const TypeWeight *>(
                b_to_use->buffer() + b_to_use->info()->offset_first_element_in_bytes());
            const int multi_stride_b = b_to_use->info()->strides_in_bytes().z() / b_to_use->info()->element_size();

            CpuAuxTensorHandler pretranspose(offset_int_vec(Pretranspose), _pretranspose_info, tensors, false);

            ARM_COMPUTE_ERROR_ON(pretranspose.get()->buffer() == nullptr);

            const bool kernel_supports_transpose = _gemm_kernel_asm->B_pretranspose_supports_transpose();
            run_parallel_pretranspose_B_array<TypeInput, TypeWeight, TypeOutput>(
                _gemm_kernel_asm.get(), pretranspose.get(), in1_ptr, ldb, multi_stride_b,
                NEScheduler::get().num_threads(), _B_pre_pretranspose_required && kernel_supports_transpose);

            b->mark_as_unused();
            // Note that we don't need to mark b_to_use as unused, as if it's been assigned to pre_pretransposed_b,
            // its memory will be auto-managed by the handler
        }

        if (_gemm_info.method == AsmConvMethod::Indirect)
        {
            prepare_indirect_buffer(tensors);
        }

        _is_prepared = true;
    }
}

template <typename TypeInput, typename TypeWeight, typename TypeOutput, class OutputStage>
bool Fallback<TypeInput, TypeWeight, TypeOutput, OutputStage>::is_configured() const
{
    return _optimised_kernel != nullptr;
}

template <typename TypeInput, typename TypeWeight, typename TypeOutput, class OutputStage>
experimental::MemoryRequirements Fallback<TypeInput, TypeWeight, TypeOutput, OutputStage>::workspace() const
{
    return _aux_mem;
}

template <typename TypeInput, typename TypeWeight, typename TypeOutput, class OutputStage>
void Fallback<TypeInput, TypeWeight, TypeOutput, OutputStage>::run(ITensorPack &tensors)
{
    auto a = tensors.get_const_tensor(TensorType::ACL_SRC_0);
    auto b = tensors.get_const_tensor(TensorType::ACL_SRC_1);
    auto c = tensors.get_const_tensor(TensorType::ACL_SRC_2);
    auto d = tensors.get_tensor(TensorType::ACL_DST);
    ARM_COMPUTE_ERROR_ON_NULLPTR(a, d);

    // Only update at runtime if the src quantization is dynamic
    if (std::is_same<OutputStage, arm_gemm::DequantizeFloat>::value &&
        (a->info()->quantization_info().is_dynamic() || b->info()->quantization_info().is_dynamic()))
    {
        // Output dequantization is just the two src scales multiplied together
        _gemm_kernel_asm->set_dequantize_scale(a->info()->quantization_info().uniform().scale *
                                               b->info()->quantization_info().uniform().scale);
    }

    int       lda = a->info()->strides_in_bytes().y() / a->info()->element_size();
    int       ldb = 0;
    const int ldd = d->info()->strides_in_bytes().y() / d->info()->element_size();

    const size_t a_batch_idx = _gemm_info.reinterpret_input_as_3d != 0 ? 3 : 2;
    const size_t a_multi_idx = a_batch_idx + 1;
    const size_t d_batch_idx = _gemm_info.depth_output_gemm3d != 0 ? 3 : 2;
    const size_t d_multi_idx = d_batch_idx + 1;

    int       batch_stride_a = a->info()->strides_in_bytes()[a_batch_idx] / a->info()->element_size();
    const int batch_stride_d = d->info()->strides_in_bytes()[d_batch_idx] / d->info()->element_size();

    int       multi_stride_a = a->info()->strides_in_bytes()[a_multi_idx] / a->info()->element_size();
    int       multi_stride_b = 0;
    const int multi_stride_d = d->info()->strides_in_bytes()[d_multi_idx] / d->info()->element_size();

    auto in0_ptr = reinterpret_cast<const TypeInput *>(a->buffer() + a->info()->offset_first_element_in_bytes());
    const TypeWeight *in1_ptr = nullptr;
    auto out_ptr = reinterpret_cast<TypeOutput *>(d->buffer() + d->info()->offset_first_element_in_bytes());

    const ITensor *b_to_use = b;

    // Pre-pretranspose B if required
    CpuAuxTensorHandler pre_pretransposed_b(
        offset_int_vec(PrePretransposedB), _pre_pretransposed_b_info, tensors,
        false /*pack_inject: no need to inject into tensors*/,
        !_run_pre_pretranspose_b /*bypass_alloc: no need to allocate if pre-pretranspose B is not required as this handle will not be used*/);
    if (b_to_use && !_is_b_constant && _run_pre_pretranspose_b)
    {
        ARM_COMPUTE_ERROR_ON(_pre_pretranspose_b == nullptr);
        ITensorPack pre_pretranspose_pack{{ACL_SRC, b_to_use}, {ACL_DST, pre_pretransposed_b.get()}};
        _pre_pretranspose_b->run(pre_pretranspose_pack);
        b_to_use = pre_pretransposed_b.get();
    }

    // Check if B is pre-tranposed and de-reference if not
    if (b_to_use && !_gemm_kernel_asm->B_is_pretransposed())
    {
        ldb            = b_to_use->info()->strides_in_bytes().y() / b_to_use->info()->element_size();
        multi_stride_b = b_to_use->info()->strides_in_bytes().z() / b_to_use->info()->element_size();
        in1_ptr        = reinterpret_cast<const TypeWeight *>(b_to_use->buffer() +
                                                       b_to_use->info()->offset_first_element_in_bytes());
    }

    // If necessary, run pretranspose every time if either weights or biases are non-constant
    if ((b_to_use && !_is_b_constant) || (c && !_is_c_constant && c->info()->data_type() == DataType::S32))
    {
        if (c && c->info()->data_type() == DataType::S32)
        {
            _gemm_kernel_asm->set_quantized_bias(
                reinterpret_cast<const int32_t *>(c->buffer() + c->info()->offset_first_element_in_bytes()), 0);
        }

        // Pretranspose B if required
        if (b_to_use && _B_pretranspose_required)
        {
            // Fixed format kernels need no pretranspose.
            ARM_COMPUTE_ERROR_ON(arm_compute::is_fixed_format(
                assembly_utils::map_to_arm_compute_weight_format(_gemm_kernel_asm->get_config().weight_format)));
            const int  ldb            = b_to_use->info()->strides_in_bytes().y() / b_to_use->info()->element_size();
            const auto b_ptr          = reinterpret_cast<const TypeWeight *>(b_to_use->buffer() +
                                                                    b_to_use->info()->offset_first_element_in_bytes());
            const int  multi_stride_b = b_to_use->info()->strides_in_bytes().z() / b_to_use->info()->element_size();

            CpuAuxTensorHandler pretranspose(offset_int_vec(Pretranspose), _pretranspose_info, tensors, true);
            ARM_COMPUTE_ERROR_ON(pretranspose.get()->buffer() == nullptr);

            if (_is_b_constant)
            {
                _gemm_kernel_asm->requantize_bias(pretranspose.get()->buffer(), b_ptr, ldb, multi_stride_b);
            }
            else
            {
                const bool kernel_supports_transpose = _gemm_kernel_asm->B_pretranspose_supports_transpose();
                run_parallel_pretranspose_B_array<TypeInput, TypeWeight, TypeOutput>(
                    _gemm_kernel_asm.get(), pretranspose.get(), b_ptr, ldb, multi_stride_b,
                    NEScheduler::get().num_threads(), _B_pre_pretranspose_required && kernel_supports_transpose);
            }
        }
    }

    // The scheduling_hint needs to be compatible with the window exposed by arm_gemm
    // The default case is when we split among the X dimension
    IScheduler::Hints scheduling_hint = IScheduler::Hints(Window::DimX);
    // If arm_gemm exposes a 2D window, perform 2D scheduling
    if (_optimised_kernel->window().num_iterations(Window::DimY) > 1 &&
        _optimised_kernel->window().num_iterations(Window::DimX) > 1)
    {
        scheduling_hint = IScheduler::Hints(IScheduler::split_dimensions_all);
    }
    // Split among Y
    else if (_optimised_kernel->window().num_iterations(Window::DimY) > 1)
    {
        scheduling_hint = IScheduler::Hints(Window::DimY);
    }

    // Set workspace if needed and reset number of threads as buffer manager gets re-created with max_threads
    CpuAuxTensorHandler workspace(offset_int_vec(AsmGemmWorkspace), _workspace_info, tensors, false);
    if (workspace.get()->buffer() != nullptr)
    {
        _gemm_kernel_asm->set_working_space(reinterpret_cast<void *>(workspace.get()->buffer()));
        const unsigned int split_dim   = scheduling_hint.split_dimension();
        const unsigned int window_size = _gemm_kernel_asm->get_window_size().total_size();
        unsigned int       num_threads = NEScheduler::get().num_threads();
        if (window_size < num_threads)
        {
            num_threads = window_size;
        }
        if (split_dim != IScheduler::split_dimensions_all)
        {
            // Make sure the kernel does not expect more threads than we can actually spawn
            const unsigned int num_iterations = _optimised_kernel->window().num_iterations(split_dim);
            num_threads                       = std::min(num_iterations, num_threads);
        }
        _gemm_kernel_asm->set_nthreads(num_threads);
    }

    // Prepare assembly kernel
    prepare(tensors);

    // Setup up matrix bias in the assembly kernel, it's just a pointer to matrix C.
    TypeOutput *bias = nullptr;
    if (c && c->info()->data_type() != DataType::S32)
    {
        bias = reinterpret_cast<TypeOutput *>(c->buffer() + c->info()->offset_first_element_in_bytes());
    }

    if (_gemm_info.method == AsmConvMethod::Indirect)
    {
        in0_ptr        = nullptr;
        lda            = 0;
        batch_stride_a = 0;
        multi_stride_a = 0;
    }

    // Set gemm parameters
    _gemm_kernel_asm->set_arrays(in0_ptr, lda, batch_stride_a, multi_stride_a, in1_ptr, ldb, multi_stride_b, out_ptr,
                                 ldd, batch_stride_d, multi_stride_d, bias, 0);

    // Need to pack the input/output pointers separately to use the thread-safe,
    // stateless-execution interface for fixed-format kernels.
    if (_gemm_info.fixed_format)
    {
        Tensor in0_tensor;
        in0_tensor.allocator()->init(*(a->info()));
        in0_tensor.allocator()->import_memory(const_cast<TypeInput *>(in0_ptr));

        Tensor in1_tensor;
        if (b)
        {
            in1_tensor.allocator()->init(*(b->info()));
            in1_tensor.allocator()->import_memory(const_cast<TypeWeight *>(in1_ptr));
        }

        Tensor bias_tensor;
        if (c)
        {
            bias_tensor.allocator()->init(*(c->info()));
            bias_tensor.allocator()->import_memory(bias);
        }

        Tensor out_tensor;
        out_tensor.allocator()->init(*(d->info()));
        out_tensor.allocator()->import_memory(out_ptr);

        ITensorPack gemm_pack{{ACL_SRC_0, &in0_tensor},
                              {ACL_SRC_1, &in1_tensor},
                              {ACL_SRC_2, &bias_tensor},
                              {ACL_SRC_3, workspace.get()},
                              {ACL_DST, &out_tensor}};

        // Schedule thread-safe stateless execution
        NEScheduler::get().schedule_op(_optimised_kernel.get(), scheduling_hint, _optimised_kernel->window(),
                                       gemm_pack);

        return;
    }

    // Schedule
    NEScheduler::get().schedule(_optimised_kernel.get(), scheduling_hint);
}

template <typename TypeInput, typename TypeWeight, typename TypeOutput>
void create_arm_gemm(std::unique_ptr<CpuGemmAssemblyDispatch::IFallback> &arm_gemm,
                     const ITensorInfo                                   *a,
                     const ITensorInfo                                   *b,
                     const ITensorInfo                                   *c,
                     ITensorInfo                                         *d,
                     arm_gemm::Activation                                 activation,
                     const AsmGemmInfo                                   &info)
{
    Params         p           = extract_parameters(a, b, d, info);
    const CPUInfo &ci          = NEScheduler::get().cpu_info();
    unsigned int   num_threads = NEScheduler::get().num_threads();

    // If fast_mode is disabled, we must enable it when fp32 accumulation is not set for fp16.
    bool is_fp16 =
        a->data_type() == DataType::F16 && b->data_type() == DataType::F16 && d->data_type() == DataType::F16;
    bool fast_mode = info.fast_mode || (is_fp16 && !info.use_fp32_acc);

    arm_gemm::GemmConfig cfg;
    cfg.weight_format = assembly_utils::map_to_arm_gemm_weight_format(info.weight_format);
    arm_gemm::GemmArgs args(&ci, p.M, p.N, p.K, p.sections, p.batches, p.multis, p.indirect, activation, num_threads,
                            info.fixed_format, fast_mode, info.accumulate, &cfg);

    // Create arm_gemm fallback
    auto fallback = std::make_unique<Fallback<TypeInput, TypeWeight, TypeOutput>>();
    fallback->configure(a, b, c, d, args, info);
    arm_gemm = std::move(fallback);
}

template <typename TypeInput, typename TypeWeight, typename TypeOutput>
void create_arm_gemm_dequant(std::unique_ptr<CpuGemmAssemblyDispatch::IFallback> &arm_gemm,
                             const ITensorInfo                                   *a,
                             const ITensorInfo                                   *b,
                             const ITensorInfo                                   *c,
                             ITensorInfo                                         *d,
                             arm_gemm::Activation                                 activation,
                             const AsmGemmInfo                                   &info)
{
    ARM_COMPUTE_UNUSED(activation);

    Params             p           = extract_parameters(a, b, d, info);
    const CPUInfo     &ci          = NEScheduler::get().cpu_info();
    const unsigned int num_threads = NEScheduler::get().num_threads();

    arm_gemm::GemmConfig cfg;
    cfg.weight_format = assembly_utils::map_to_arm_gemm_weight_format(info.weight_format);
    arm_gemm::GemmArgs args(&ci, p.M, p.N, p.K, p.sections, p.batches, p.multis, p.indirect, activation, num_threads,
                            info.fixed_format, info.fast_mode, info.accumulate, &cfg);

    // Create arm_gemm fallback
    auto fallback = std::make_unique<Fallback<TypeInput, TypeWeight, TypeOutput, arm_gemm::DequantizeFloat>>();

    // Configure requantization info
    const GEMMLowpOutputStageInfo os_info = info.output_stage;

    arm_gemm::DequantizeFloat gemm_dequant_info{};
    gemm_dequant_info = arm_gemm::DequantizeFloat(d->quantization_info().uniform().scale);

    fallback->configure(a, b, c, d, args, info, gemm_dequant_info);
    arm_gemm = std::move(fallback);
}

template <typename TypeInput, typename TypeWeight, typename TypeOutput>
void create_arm_gemm_quant(std::unique_ptr<CpuGemmAssemblyDispatch::IFallback> &arm_gemm,
                           const ITensorInfo                                   *a,
                           const ITensorInfo                                   *b,
                           const ITensorInfo                                   *c,
                           ITensorInfo                                         *d,
                           arm_gemm::Activation                                 activation,
                           const AsmGemmInfo                                   &info)
{
    ARM_COMPUTE_UNUSED(activation);
    Params             p           = extract_parameters(a, b, d, info);
    const CPUInfo     &ci          = NEScheduler::get().cpu_info();
    const unsigned int num_threads = NEScheduler::get().num_threads();

    arm_gemm::GemmConfig cfg;
    cfg.weight_format = assembly_utils::map_to_arm_gemm_weight_format(info.weight_format);
    arm_gemm::GemmArgs args(&ci, p.M, p.N, p.K, p.sections, p.batches, p.multis, p.indirect, activation, num_threads,
                            info.fixed_format, info.fast_mode, info.accumulate, &cfg);

    // Create arm_gemm fallback
    auto fallback = std::make_unique<Fallback<TypeInput, TypeWeight, TypeOutput, arm_gemm::Requantize32>>();

    // Configure requantization info
    const int32_t                 negation = info.negated_offsets ? 1 : -1;
    const int32_t                 a_offset = -a->quantization_info().uniform().offset * negation;
    const int32_t                 b_offset = -b->quantization_info().uniform().offset * negation;
    const GEMMLowpOutputStageInfo os_info  = info.output_stage;

    arm_gemm::Requantize32 gemm_requant_info{};
    if (os_info.gemmlowp_shifts.size() > 1)
    {
        const auto requantize_data =
            fallback->set_requantize_data(os_info.gemmlowp_shifts, os_info.gemmlowp_multipliers);
        gemm_requant_info = arm_gemm::Requantize32(
            nullptr, 0, a_offset, b_offset, os_info.gemmlowp_offset,
            (std::get<0>(requantize_data)) ? std::get<1>(requantize_data) : nullptr, std::get<2>(requantize_data),
            std::get<3>(requantize_data), os_info.gemmlowp_min_bound, os_info.gemmlowp_max_bound);
    }
    else
    {
        gemm_requant_info =
            arm_gemm::Requantize32(nullptr, 0, a_offset, b_offset, os_info.gemmlowp_offset, -os_info.gemmlowp_shift,
                                   os_info.gemmlowp_multiplier, os_info.gemmlowp_min_bound, os_info.gemmlowp_max_bound);
    }

    // Configure fallback
    fallback->configure(a, b, c, d, args, info, gemm_requant_info);
    arm_gemm = std::move(fallback);
}
} //namespace

CpuGemmAssemblyDispatch::CpuGemmAssemblyDispatch() : _arm_gemm(nullptr)
{
}

Status CpuGemmAssemblyDispatch::has_opt_impl(arm_compute::WeightFormat &expected_weight_format,
                                             const ITensorInfo         *a,
                                             const ITensorInfo         *b,
                                             const ITensorInfo         *c,
                                             const ITensorInfo         *d,
                                             const AsmGemmInfo         &info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(a, b, d);
    ARM_COMPUTE_UNUSED(c);
    arm_gemm::Activation act         = assembly_utils::map_to_arm_gemm_activation(info.activation_info);
    Params               p           = extract_parameters(a, b, d, info);
    const CPUInfo       &ci          = NEScheduler::get().cpu_info();
    unsigned int         num_threads = NEScheduler::get().num_threads();

    // If fast_mode is disabled, we must enable it when fp32 accumulation is not set for fp16.
    bool is_fp16 =
        a->data_type() == DataType::F16 && b->data_type() == DataType::F16 && d->data_type() == DataType::F16;
    bool fast_mode = info.fast_mode || (is_fp16 && !info.use_fp32_acc);

    arm_gemm::GemmConfig cfg;
    cfg.weight_format                           = assembly_utils::map_to_arm_gemm_weight_format(info.weight_format);
    arm_gemm::WeightFormat arm_gemm_expected_wf = assembly_utils::map_to_arm_gemm_weight_format(expected_weight_format);
    arm_gemm::GemmArgs     args(&ci, p.M, p.N, p.K, p.sections, p.batches, p.multis, p.indirect, act, num_threads,
                                info.fixed_format, fast_mode, info.accumulate, &cfg);
    // TODO(COMPMID-6595): Incorporate info.transpose_b
    switch (a->data_type())
    {
        case DataType::F32:
            ARM_COMPUTE_RETURN_ERROR_ON_MSG(
                !(arm_gemm::has_opt_gemm<float, float, float, arm_gemm::Nothing>(arm_gemm_expected_wf, args, {})),
                "We could not find an optimized kernel for F32 input");
            break;
#ifdef __aarch64__
        case DataType::U8:
        case DataType::QASYMM8:
            if (d->data_type() == DataType::S32)
            {
                ARM_COMPUTE_RETURN_ERROR_ON_MSG(
                    !(arm_gemm::has_opt_gemm<uint8_t, uint8_t, uint32_t, arm_gemm::Nothing>(arm_gemm_expected_wf, args,
                                                                                            {})),
                    "We could not find an optimized kernel for U8/QASYMM8 input and U32 output");
            }
            else if (b->data_type() == DataType::QASYMM8_SIGNED)
            {
                ARM_COMPUTE_RETURN_ERROR_ON_MSG(
                    !(arm_gemm::has_opt_gemm<uint8_t, int8_t, uint8_t, arm_gemm::Requantize32>(arm_gemm_expected_wf,
                                                                                               args, {})),
                    "We could not find an optimized kernel for U8 input with S8 weights and U8 output");
            }
            else
            {
                ARM_COMPUTE_RETURN_ERROR_ON_MSG(
                    !(arm_gemm::has_opt_gemm<uint8_t, uint8_t, uint8_t, arm_gemm::Requantize32>(arm_gemm_expected_wf,
                                                                                                args, {})),
                    "We could not find an optimized kernel for U8 input and U8 output");
            }
            break;
        case DataType::S8:
        case DataType::QASYMM8_SIGNED:
            if (d->data_type() == DataType::S32)
            {
                ARM_COMPUTE_RETURN_ERROR_ON_MSG(
                    !(arm_gemm::has_opt_gemm<int8_t, int8_t, int32_t, arm_gemm::Nothing>(arm_gemm_expected_wf, args,
                                                                                         {})),
                    "We could not find an optimized kernel for S8/QASYMM8_SIGNED input and S32 output");
            }
            else
            {
                ARM_COMPUTE_RETURN_ERROR_ON_MSG(
                    !(arm_gemm::has_opt_gemm<int8_t, int8_t, int8_t, arm_gemm::Requantize32>(arm_gemm_expected_wf, args,
                                                                                             {})),
                    "We could not find an optimized kernel for S8 input and S8 output");
            }
            break;
#endif /* __aarch64__ */

#if defined(ARM_COMPUTE_ENABLE_BF16)
        case DataType::BFLOAT16:
        {
            if (d->data_type() == DataType::BFLOAT16)
            {
                ARM_COMPUTE_RETURN_ERROR_ON_MSG(
                    !(arm_gemm::has_opt_gemm<bfloat16, bfloat16, bfloat16, arm_gemm::Nothing>(arm_gemm_expected_wf,
                                                                                              args, {})),
                    "We could not find an optimized kernel for BFLOAT16 input and BFLOAT16 output");
            }
            else
            {
                ARM_COMPUTE_RETURN_ERROR_ON_MSG(
                    !(arm_gemm::has_opt_gemm<bfloat16, bfloat16, float, arm_gemm::Nothing>(arm_gemm_expected_wf, args,
                                                                                           {})),
                    "We could not find an optimized kernel for BFLOAT16 input and F32 output");
            }
            break;
        }
#endif /* defined(ARM_COMPUTE_ENABLE_BF16) */

#if defined(ENABLE_FP16_KERNELS)
        case DataType::F16:
            if (d->data_type() == DataType::F16)
            {
                ARM_COMPUTE_RETURN_ERROR_ON_MSG(
                    !(arm_gemm::has_opt_gemm<float16_t, float16_t, float16_t, arm_gemm::Nothing>(arm_gemm_expected_wf,
                                                                                                 args, {})),
                    "We could not find an optimized kernel for F16 input and F16 output");
            }
            else
            {
                ARM_COMPUTE_RETURN_ERROR_ON_MSG(
                    !(arm_gemm::has_opt_gemm<float16_t, float16_t, float, arm_gemm::Nothing>(arm_gemm_expected_wf, args,
                                                                                             {})),
                    "We could not find an optimized kernel for F16 input and F32 output");
            }
            break;
#endif /* ENABLE_FP16_KERNELS */
        default:
            ARM_COMPUTE_RETURN_ERROR_ON_MSG(true, "Unsupported type. Could not find a kernel");
            break;
    }
    expected_weight_format = assembly_utils::map_to_arm_compute_weight_format(arm_gemm_expected_wf);

    return Status{};
}

Status CpuGemmAssemblyDispatch::validate(
    const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, const ITensorInfo *d, const AsmGemmInfo &info)
{
    ARM_COMPUTE_UNUSED(c, info);
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(a, b, d);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(a);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(d);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_BF16_UNSUPPORTED(a);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(!(info.reshape_b_only_on_first_run),
                                    "Assembly kernel will not be executed when reshape_b_only_on_first_run is false");

#ifndef __aarch64__
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(a->element_size() == 1, "8bit integer types only supported for aarch64");
#endif /* __aarch64__ */
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(a, 1, DataType::U8, DataType::QASYMM8,
                                                         DataType::QASYMM8_SIGNED, DataType::S8, DataType::BFLOAT16,
                                                         DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(
        b, 1, DataType::U8, DataType::QASYMM8, DataType::QASYMM8_SIGNED, DataType::QSYMM8_PER_CHANNEL, DataType::S8,
        DataType::BFLOAT16, DataType::F16, DataType::F32);

    if (is_data_type_quantized_per_channel(b->data_type()))
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(a, 1, DataType::QASYMM8_SIGNED, DataType::S8);
    }
    else if (is_fixed_format_fast_math(info.weight_format))
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN(a, DataType::F32);
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN(b, DataType::BFLOAT16);
    }
    else if (!(a->data_type() == DataType::QASYMM8 && b->data_type() == DataType::QASYMM8_SIGNED))
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(a, b);
    }
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(a->data_type() == DataType::F32 && d->data_type() != DataType::F32,
                                    "Only F32 output supported for F32 input");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(a->data_type() == DataType::F16 &&
                                        (d->data_type() != DataType::F32 && d->data_type() != DataType::F16),
                                    "Only F32/F16 output supported for F16 input");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(a->data_type() == DataType::BFLOAT16 &&
                                        (d->data_type() != DataType::F32 && d->data_type() != DataType::BFLOAT16),
                                    "Only F32/BFLOAT16 output supported for BFLOAT16 input");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(a->data_type() == DataType::U8 && d->data_type() != DataType::U32,
                                    "Only U32 output supported for U8 input");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(a->data_type() == DataType::S8 && d->data_type() != DataType::S32,
                                    "Only S32 output supported for S8 input");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(
        a->data_type() == DataType::QASYMM8 &&
            (d->data_type() != DataType::QASYMM8 && d->data_type() != DataType::S32 && d->data_type() != DataType::F32),
        "Only QASYMM8/S32/F32 output supported for QASYMM8 input");
    arm_compute::WeightFormat expected_weight_format = arm_compute::WeightFormat::UNSPECIFIED;
    const Status              ret = CpuGemmAssemblyDispatch::has_opt_impl(expected_weight_format, a, b, c, d, info);
    if (bool(ret) && expected_weight_format != arm_compute::WeightFormat::ANY)
    {
        // Correctness check: if the format expected by the kernel is
        // not "any", make sure that the one found matches the format
        // intended by the caller.
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(
            (expected_weight_format != info.weight_format),
            "The format expected by the kernel does not correspond with the one requested by the user.");
    }
    return ret;
}

bool CpuGemmAssemblyDispatch::is_activation_supported(const ActivationLayerInfo &activation)
{
    arm_gemm::Activation act = assembly_utils::map_to_arm_gemm_activation(activation);
    return act.type != arm_gemm::Activation::Type::None;
}

void CpuGemmAssemblyDispatch::configure(
    const ITensorInfo *a, const ITensorInfo *b, const ITensorInfo *c, ITensorInfo *d, const AsmGemmInfo &info)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(a, b, d);
    arm_gemm::Activation act = assembly_utils::map_to_arm_gemm_activation(info.activation_info);

    //If we don't support a combination of data types, silently return: it is the caller's responsibility to check if configure() was successful via is_configured()
    if (!CpuGemmAssemblyDispatch::validate(a, b, c, d, info))
    {
        return;
    }

    switch (a->data_type())
    {
        case DataType::F32:
            create_arm_gemm<float, float, float>(_arm_gemm, a, b, c, d, act, info);
            break;
#ifdef __aarch64__
        case DataType::U8:
        case DataType::QASYMM8:
            if (b->data_type() == DataType::S8 || b->data_type() == DataType::QASYMM8_SIGNED)
            {
                if (d->data_type() == DataType::F32)
                {
                    create_arm_gemm_dequant<uint8_t, int8_t, float>(_arm_gemm, a, b, c, d, act, info);
                }
                else
                {
                    create_arm_gemm_quant<uint8_t, int8_t, uint8_t>(_arm_gemm, a, b, c, d, act, info);
                }
            }
            else if (d->data_type() == DataType::S32)
            {
                create_arm_gemm<uint8_t, uint8_t, uint32_t>(_arm_gemm, a, b, c, d, act, info);
            }
            else
            {
                create_arm_gemm_quant<uint8_t, uint8_t, uint8_t>(_arm_gemm, a, b, c, d, act, info);
            }
            break;
        case DataType::S8:
        case DataType::QASYMM8_SIGNED:
            if (d->data_type() == DataType::S32)
            {
                create_arm_gemm<int8_t, int8_t, int32_t>(_arm_gemm, a, b, c, d, act, info);
            }
            else if (d->data_type() == DataType::F32)
            {
                create_arm_gemm_dequant<int8_t, int8_t, float>(_arm_gemm, a, b, c, d, act, info);
            }
#if defined(ENABLE_FP16_KERNELS)
            else if (d->data_type() == DataType::F16)
            {
                create_arm_gemm_dequant<int8_t, int8_t, float16_t>(_arm_gemm, a, b, c, d, act, info);
            }
#endif /* defined(ENABLE_FP16_KERNELS) */
            else
            {
                create_arm_gemm_quant<int8_t, int8_t, int8_t>(_arm_gemm, a, b, c, d, act, info);
            }
            break;
#endif /* __aarch64__ */
#if defined(ARM_COMPUTE_ENABLE_BF16)
        case DataType::BFLOAT16:
            if (d->data_type() == DataType::BFLOAT16)
            {
                create_arm_gemm<bfloat16, bfloat16, bfloat16>(_arm_gemm, a, b, c, d, act, info);
            }
            else
            {
                create_arm_gemm<bfloat16, bfloat16, float>(_arm_gemm, a, b, c, d, act, info);
            }
            break;
#endif /* defined(ARM_COMPUTE_ENABLE_BF16) */
#ifdef ENABLE_FP16_KERNELS
        case DataType::F16:
            if (d->data_type() == DataType::F16)
            {
                create_arm_gemm<float16_t, float16_t, float16_t>(_arm_gemm, a, b, c, d, act, info);
            }
            else
            {
                create_arm_gemm<float16_t, float16_t, float>(_arm_gemm, a, b, c, d, act, info);
            }
            break;
#endif /* ENABLE_FP16_KERNELS */
        default:
            break;
    }
}

void CpuGemmAssemblyDispatch::prepare(ITensorPack &tensors)
{
    ARM_COMPUTE_ERROR_ON(_arm_gemm == nullptr);
    _arm_gemm->prepare(tensors);
}

bool CpuGemmAssemblyDispatch::is_configured() const
{
    return _arm_gemm && _arm_gemm->is_configured();
}

void CpuGemmAssemblyDispatch::run(ITensorPack &tensors)
{
    ARM_COMPUTE_ERROR_ON(_arm_gemm == nullptr);
    _arm_gemm->run(tensors);
}

experimental::MemoryRequirements CpuGemmAssemblyDispatch::workspace() const
{
    ARM_COMPUTE_ERROR_ON(_arm_gemm == nullptr);
    return _arm_gemm->workspace();
}

void CpuGemmAssemblyDispatch::update_quantization_parameters(const GEMMLowpOutputStageInfo &output_info,
                                                             const QuantizationInfo        &a,
                                                             const QuantizationInfo        &b,
                                                             const bool                     is_prepared,
                                                             const bool                     negated_offsets)
{
    ARM_COMPUTE_ERROR_ON(_arm_gemm == nullptr);
    _arm_gemm->update_quantization_parameters(output_info, a, b, is_prepared, negated_offsets);
}
} // namespace cpu
} // namespace arm_compute
