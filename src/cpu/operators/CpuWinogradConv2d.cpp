/*
 * Copyright (c) 2021-2022 Arm Limited.
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
#include "src/cpu/operators/CpuWinogradConv2d.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/utils/quantization/AsymmHelpers.h"
#include "arm_compute/runtime/FunctionDescriptors.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "src/common/utils/Log.h"
#include "src/core/CPP/Validate.h"
#include "src/core/NEON/kernels/assembly/winograd.hpp"
#include "src/core/NEON/kernels/convolution/common/tensor.hpp"
#include "src/core/NEON/kernels/convolution/common/utils.hpp"
#include "src/core/helpers/MemoryHelpers.h"
#include "src/core/helpers/WindowHelpers.h"
#include "src/core/utils/AssemblyUtils.h"
#include "src/cpu/kernels/CpuWinogradConv2dKernel.h"
#include "src/cpu/kernels/assembly/arm_gemm.hpp"
#include "src/cpu/operators/CpuActivation.h"
#include "src/cpu/operators/CpuPermute.h"
#include "src/cpu/utils/CpuAuxTensorHandler.h"
#include "support/Cast.h"

namespace arm_compute
{
namespace cpu
{
using namespace arm_compute::experimental;
using namespace arm_compute::utils::cast;

namespace
{
inline Tensor4DShape internal_get_shape(const ITensorInfo *in)
{
    const DataLayout data_layout = in->data_layout();
    const int        in_width    = in->dimension(get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH));
    const int        in_height   = in->dimension(get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT));
    const int        in_channels = in->dimension(get_data_layout_dimension_index(data_layout, DataLayoutDimension::CHANNEL));
    const int        in_batches  = in->dimension(get_data_layout_dimension_index(data_layout, DataLayoutDimension::BATCHES));

    return Tensor4DShape{ in_batches, in_height, in_width, in_channels };
}

Status validate_arguments(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *dst, const PadStrideInfo &conv_info)
{
    ARM_COMPUTE_UNUSED(dst, weights);
    ARM_COMPUTE_RETURN_ERROR_ON_CPU_F16_UNSUPPORTED(src);

    ARM_COMPUTE_RETURN_ERROR_ON_MSG(conv_info.stride().first != 1 || conv_info.stride().second != 1, "Winograd layer only supports unit strides.");
    if(biases != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, biases);
        ARM_COMPUTE_RETURN_ERROR_ON(biases->num_dimensions() > 1);
    }
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src, 1, DataType::F16, DataType::F32);
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src, weights);
    return Status{};
}

bool get_winograd_kernel_implementation(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *dst,
                                        const PadStrideInfo &conv_info, const ActivationLayerInfo &act_info, bool enable_fast_math,
                                        arm_conv::winograd::WinogradImpl *winograd_impl, std::unique_ptr<arm_conv::ConvolutionArgs> &conv_args)
{
    arm_conv::winograd::WinogradConfig winograd_cfg;
    arm_gemm::GemmConfig               cfg;

    const DataType data_type = src->data_type();
    Tensor4DShape  in_shape{ internal_get_shape(src) };
    Tensor4DShape  out_shape{ internal_get_shape(dst) };
    Tensor4DShape  kernel_shape{ internal_get_shape(weights) };
    uint32_t       nthreads = NEScheduler::get().num_threads();
    // Get configuration arguments for Winograd
    winograd_cfg.output_rows = 0;
    winograd_cfg.output_cols = 0;
    conv_args                = std::make_unique<arm_conv::ConvolutionArgs>(
                                   in_shape.n_batches,
                                   arm_conv::Shape2D{ static_cast<uint32_t>(in_shape.n_rows), static_cast<uint32_t>(in_shape.n_cols) },
                                   in_shape.n_channels,
                                   conv_info.pad_top(),
                                   conv_info.pad_left(),
                                   arm_conv::Shape2D{ static_cast<uint32_t>(out_shape.n_rows), static_cast<uint32_t>(out_shape.n_cols) },
                                   out_shape.n_channels,
                                   arm_conv::Shape2D{ static_cast<uint32_t>(kernel_shape.n_rows), static_cast<uint32_t>(kernel_shape.n_cols) },
                                   assembly_utils::map_to_arm_gemm_activation(act_info));

    bool success = false;
    if(data_type == DataType::F32)
    {
        success = arm_conv::winograd::get_implementation<float>(
                      *winograd_impl, &CPUInfo::get(), *conv_args, nthreads, enable_fast_math, &winograd_cfg, nullptr);
    }
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    else if(data_type == DataType::F16)
    {
        success = arm_conv::winograd::get_implementation<__fp16>(
                      *winograd_impl, &CPUInfo::get(), *conv_args, nthreads, enable_fast_math, &winograd_cfg, nullptr);
    }
#endif // defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    else
    {
        success = false;
    }
    return success;
}
inline bool fuse_function_supported(const ActivationLayerInfo &act_info)
{
    return act_info.activation() == ActivationLayerInfo::ActivationFunction::RELU || act_info.activation() == ActivationLayerInfo::ActivationFunction::BOUNDED_RELU;
}
} // namespace

CpuWinogradConv2d::CpuWinogradConv2d()

    : _gemm_function(std::make_unique<CpuGemm>()),
      _activation_func(std::make_unique<CpuActivation>()),
      _transform_input_kernel(nullptr),
      _transform_output_kernel(nullptr),
      _permute_input(std::make_unique<CpuPermute>()),
      _permute_output(std::make_unique<CpuPermute>()),
      _permute_weights(std::make_unique<CpuPermute>()),
      _aux_mem(AuxTensorIdx::Count),
      _conv_args{ nullptr },
      _winograd_impl{},
      _data_layout(),
      _winograd_transformed_input{},
      _winograd_transformed_output{},
      _winograd_transformed_weights{},
      _input_workspace(),
      _output_workspace(),
      _weights_hwio(),
      _input_nhwc(),
      _output_nhwc(),
      _is_prepared{ false },
      _run_activation{ false }
{
}

CpuWinogradConv2d::~CpuWinogradConv2d() = default;

void CpuWinogradConv2d::configure(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, ITensorInfo *dst,
                                  const PadStrideInfo &conv_info, const ActivationLayerInfo &act_info, bool enable_fast_math)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src, weights, dst);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src, weights, biases, dst, conv_info));
    ARM_COMPUTE_LOG_PARAMS(src, weights, biases, dst, conv_info, act_info, enable_fast_math);
    ARM_COMPUTE_UNUSED(biases);
    const DataType data_type = src->data_type();
    uint32_t       nthreads  = NEScheduler::get().num_threads();
    _data_layout             = src->data_layout();
    const Tensor4DShape kernel_shape{ internal_get_shape(weights) };

    bool success = get_winograd_kernel_implementation(src, weights, dst, conv_info, act_info, enable_fast_math, &_winograd_impl, _conv_args);

    ARM_COMPUTE_EXIT_ON_MSG_VAR(!success, "Unsupported kernel size: %d x %d.\n", kernel_shape.n_rows, kernel_shape.n_cols);
    ARM_COMPUTE_LOG_MSG_WITH_FORMAT_ACL(arm_compute::logging::LogLevel::INFO, "Using input transform: %s\n", _winograd_impl.input_transform->get_name().c_str());
    ARM_COMPUTE_LOG_MSG_WITH_FORMAT_ACL(arm_compute::logging::LogLevel::INFO, "Using weight transform: %s\n", _winograd_impl.input_transform->get_name().c_str());
    ARM_COMPUTE_LOG_MSG_WITH_FORMAT_ACL(arm_compute::logging::LogLevel::INFO, "Using output transform: %s\n", _winograd_impl.input_transform->get_name().c_str());

    const bool has_impl = ((_winograd_impl.input_transform != nullptr) && (_winograd_impl.output_transform != nullptr) && (_winograd_impl.gemm_args != nullptr));
    if(has_impl)
    {
        // Determine how much working space is required, allocate it.
        const size_t input_workspace_size  = _winograd_impl.input_transform->get_working_space_size(*_conv_args, nthreads);
        const size_t output_workspace_size = _winograd_impl.output_transform->get_working_space_size(*_conv_args, nthreads);

        TensorInfo input_workspace_info(TensorShape(input_workspace_size), 1, DataType::U8);
        TensorInfo output_workspace_info(TensorShape(output_workspace_size), 1, DataType::U8);
        _input_workspace  = input_workspace_info;
        _output_workspace = output_workspace_info;

        const auto &wds = _winograd_impl.winograd_spec;

        // Preparing winograd transformed input tensor
        const size_t     data_type_size    = src->element_size();
        const uint32_t   m                 = _winograd_impl.gemm_args->_Msize; // Total number of tiles
        const uint32_t   k                 = _winograd_impl.gemm_args->_Ksize; // Input channels
        const uint32_t   n                 = _winograd_impl.gemm_args->_Nsize; // Output channels
        const uint32_t   n_gemms           = _winograd_impl.gemm_args->_nmulti;
        const uint32_t   n_batches         = _winograd_impl.gemm_args->_nbatches;
        constexpr size_t storage_alignment = 64;

        const TensorShape a_shape(k, m, n_batches, n_gemms);
        Strides           a_strides(data_type_size);
        a_strides.set(1, data_type_size * _winograd_impl.winograd_spec.input_ld_row);
        a_strides.set(2, data_type_size * _winograd_impl.winograd_spec.input_ld_batch);
        a_strides.set(3, data_type_size * _winograd_impl.winograd_spec.input_ld_matrix);

        const TensorShape b_shape(n, k, n_gemms);
        Strides           b_strides(data_type_size);
        b_strides.set(1, data_type_size * _winograd_impl.winograd_spec.weight_ld_row);
        b_strides.set(2, data_type_size * _winograd_impl.winograd_spec.weight_ld_matrix);

        const TensorShape d_shape(n, m, n_batches, n_gemms);
        Strides           d_strides(data_type_size);
        d_strides.set(1, data_type_size * _winograd_impl.winograd_spec.output_ld_row);
        d_strides.set(2, data_type_size * _winograd_impl.winograd_spec.output_ld_batch);
        d_strides.set(3, data_type_size * _winograd_impl.winograd_spec.output_ld_matrix);

        TensorInfo a_info{};
        TensorInfo b_info{};
        TensorInfo d_info{};
        a_info.init(a_shape, 1, data_type, a_strides, 0, wds.input_matrix_size_bytes);
        b_info.init(b_shape, 1, data_type, b_strides, 0, wds.weight_matrix_size_bytes);
        d_info.init(d_shape, 1, data_type, d_strides, 0, wds.output_matrix_size_bytes);

        _winograd_transformed_input   = a_info;
        _winograd_transformed_weights = b_info;
        _winograd_transformed_output  = d_info;

        PermutationVector weights_permutation_vector(3U, 0U, 1U, 2U);

        // Configure the kernel to transform the input tensor from NCHW -> NHWC
        if(_data_layout == DataLayout::NCHW)
        {
            _permute_input->configure(src, &_input_nhwc, PermutationVector(2U, 0U, 1U));
            weights_permutation_vector = PermutationVector(3U, 2U, 0U, 1U);
        }

        // Re-order a weight tensor from [Output feature map x Input feature map x Height x Width] to [Height x Width x Input feature map x Output feature map]
        _permute_weights->configure(weights, &_weights_hwio, weights_permutation_vector);

        // Reorder the convoluted output to ACL's ordering NCHW
        if(_data_layout == DataLayout::NCHW)
        {
            // configure and allocate dst tensor to be used to convert from winograd domain to spatial domain when calling to reshape_output()
            TensorInfo info(TensorShape(dst->dimension(2), dst->dimension(0),
                                        dst->dimension(1), dst->dimension(3)),
                            1, dst->data_type());
            _output_nhwc = info;
            _permute_output->configure(&_output_nhwc, dst, PermutationVector(1U, 2U, 0U));
        }

        // Configure GEMM function
        _gemm_function->configure(&_winograd_transformed_input, &_winograd_transformed_weights, nullptr, &_winograd_transformed_output, 1.0f, 0.f);

        //Configure Activation Layer
        _run_activation = act_info.enabled() && !fuse_function_supported(act_info);
        if(_run_activation)
        {
            _activation_func->configure(dst, nullptr, act_info);
        }

        auto asm_mem_req         = _gemm_function->workspace();
        _aux_mem[GemmWorkspace]  = asm_mem_req[GemmWorkspace];
        _aux_mem[Pretranspose]   = asm_mem_req[Pretranspose];
        _aux_mem[InterleavedLHS] = asm_mem_req[InterleavedLHS];
        _aux_mem[TransposedRHS]  = asm_mem_req[TransposedRHS];
        _aux_mem[TempResult]     = asm_mem_req[TempResult];

        // Request temporary memory. Overlap memory needed for Input/Output transformations as they run on different non-overlapping time-steps.
        _aux_mem[TransformedInput]   = MemoryInfo(offset_int_vec(TransformedInput), MemoryLifetime::Temporary, wds.input_matrix_size_bytes, storage_alignment);
        _aux_mem[TransformedOutput]  = MemoryInfo(offset_int_vec(TransformedOutput), MemoryLifetime::Temporary, wds.output_matrix_size_bytes, storage_alignment);
        _aux_mem[WorkspaceIO]        = MemoryInfo(offset_int_vec(WorkspaceIO), MemoryLifetime::Temporary, std::max(input_workspace_size, output_workspace_size));
        _aux_mem[PermutedWeights]    = MemoryInfo(offset_int_vec(PermutedWeights), MemoryLifetime::Prepare, _weights_hwio.total_size());
        _aux_mem[TransformedWeights] = MemoryInfo(offset_int_vec(TransformedWeights), MemoryLifetime::Persistent, wds.weight_matrix_size_bytes, storage_alignment);
        if(_data_layout == DataLayout::NCHW)
        {
            _aux_mem[PermutedInput].merge(offset_int_vec(PermutedInput), src->total_size());
            _aux_mem[PermutedOutput].merge(offset_int_vec(PermutedOutput), dst->total_size());
        }
    }
}
Status CpuWinogradConv2d::validate(const ITensorInfo *src, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *dst,
                                   const PadStrideInfo &conv_info, const ActivationLayerInfo &act_info, bool enable_fast_math)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src, weights, dst);
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src, weights, biases, dst, conv_info));

    const Tensor4DShape              kernel_shape{ internal_get_shape(weights) };
    arm_conv::winograd::WinogradImpl winograd_impl{};

    std::unique_ptr<arm_conv::ConvolutionArgs> conv_args;
    const bool                                 success = get_winograd_kernel_implementation(src, weights, dst, conv_info, act_info, enable_fast_math, &winograd_impl, conv_args);

    ARM_COMPUTE_RETURN_ERROR_ON_MSG_VAR(success == false, "Unsupported kernel size: %d x %d.\n", kernel_shape.n_rows, kernel_shape.n_cols);
    ARM_COMPUTE_LOG_MSG_WITH_FORMAT_ACL(arm_compute::logging::LogLevel::INFO, "Using input transform: %s\n", winograd_impl.input_transform->get_name().c_str());
    ARM_COMPUTE_LOG_MSG_WITH_FORMAT_ACL(arm_compute::logging::LogLevel::INFO, "Using weight transform: %s\n", winograd_impl.input_transform->get_name().c_str());
    ARM_COMPUTE_LOG_MSG_WITH_FORMAT_ACL(arm_compute::logging::LogLevel::INFO, "Using output transform: %s\n", winograd_impl.input_transform->get_name().c_str());
    return Status{};
}

void CpuWinogradConv2d::run(ITensorPack &tensors)
{
    prepare(tensors);
    auto   src    = tensors.get_const_tensor(ACL_SRC_0);
    auto   biases = tensors.get_const_tensor(ACL_SRC_2);
    auto   output = tensors.get_tensor(ACL_DST);
    Window win;

    const uint32_t nthreads = NEScheduler::get().num_threads();

    // The Winograd transform implementation does fine-grain threading inside the transforms. Just pass thread_id and nthreads.
    win.set(Window::DimX, Window::Dimension(0, nthreads, 1));

    // Wrap the winograd-domain tensorInfos created in configuration in tensors and allocate the required memory.
    CpuAuxTensorHandler input_nhwc(offset_int_vec(PermutedInput), _input_nhwc, tensors, true);
    CpuAuxTensorHandler winograd_input_transformed(offset_int_vec(TransformedInput), _winograd_transformed_input, tensors, true);
    CpuAuxTensorHandler input_workspace(offset_int_vec(WorkspaceIO), _input_workspace, tensors, true);
    const bool          is_nchw = _data_layout == DataLayout::NCHW;
    if(is_nchw)
    {
        //Bring channels to the front as Winograd code expects the tensor to be in the format NHWC
        ITensorPack pack{ { ACL_SRC, src }, { ACL_DST, input_nhwc.get() } };
        _permute_input->run(pack);
    }

    CpuAuxTensorHandler winograd_output_transformed(offset_int_vec(TransformedOutput), _winograd_transformed_output, tensors, true);
    CpuAuxTensorHandler output_workspace(offset_int_vec(WorkspaceIO), _output_workspace, tensors, true);
    CpuAuxTensorHandler output_nhwc(offset_int_vec(PermutedOutput), _output_nhwc, tensors, true);

    ITensorPack transform_input_pack{ { ACL_SRC, is_nchw ? input_nhwc.get() : src }, { ACL_DST, winograd_input_transformed.get() }, { ACL_INT, input_workspace.get() } };
    _transform_input_kernel = std::make_unique<CpuWinogradConv2dTransformInputKernel>(_winograd_impl, *_conv_args, nthreads);

    NEScheduler::get().schedule_op(_transform_input_kernel.get(), Window::DimX, win, transform_input_pack);

    CpuAuxTensorHandler winograd_weights_transformed(offset_int_vec(TransformedWeights), _winograd_transformed_weights, tensors, true);

    // Run 16 GEMMs in multiple threads, each kernel runs one or more GEMMs
    ITensorPack gemm_pack = tensors;
    gemm_pack.add_const_tensor(ACL_SRC, winograd_input_transformed.get());
    gemm_pack.add_const_tensor(ACL_SRC_1, winograd_weights_transformed.get());
    gemm_pack.add_const_tensor(ACL_BIAS, nullptr);
    gemm_pack.add_tensor(ACL_DST, winograd_output_transformed.get());
    _gemm_function->run(gemm_pack);

    // Output transform
    _transform_output_kernel = std::make_unique<CpuWinogradConv2dTransformOutputKernel>(_winograd_impl, *_conv_args, nthreads);
    ITensorPack transform_output_pack{ { ACL_SRC_0, winograd_output_transformed.get() }, { ACL_DST, is_nchw ? output_nhwc.get() : output }, { ACL_SRC_1, biases }, { ACL_INT, output_workspace.get() } };
    NEScheduler::get().schedule_op(_transform_output_kernel.get(), Window::DimX, win, transform_output_pack);
    if(is_nchw)
    {
        // Reorder the convoluted output to ACL's ordering NCHW
        ITensorPack pack{ { ACL_SRC, output_nhwc.get() }, { ACL_DST, output } };
        _permute_output->run(pack);
    }
    if(_run_activation)
    {
        ITensorPack pack{ { ACL_SRC, output }, { ACL_DST, output } };
        _activation_func->run(pack);
    }
}

void CpuWinogradConv2d::prepare(ITensorPack &tensors)
{
    if(!_is_prepared)
    {
        const ITensor *weights     = tensors.get_const_tensor(ACL_SRC_1);
        ITensor       *weights_aux = utils::cast::polymorphic_cast<ITensor *>(tensors.get_tensor(offset_int_vec(PermutedWeights)));

        CpuAuxTensorHandler permuted_weights(_weights_hwio, *weights_aux);
        ITensorPack         permute_tensors{ { ACL_SRC, weights }, { ACL_DST, permuted_weights.get() } };
        _permute_weights->run(permute_tensors);
        const int element_size_in_bytes = permuted_weights.get()->info()->element_size();
        // Weights were in OHWI format, before being permuted "permuted_weights" to be in HWIO format.
        const unsigned int height_idx  = 3; // H in HWIO
        const unsigned int width_idx   = 2; // W in HWIO
        const unsigned int channel_idx = 1; // I in HWIO

        const int permuted_weight_row_stride     = permuted_weights.get()->info()->strides_in_bytes()[height_idx] / element_size_in_bytes;
        const int permuted_weight_col_stride     = permuted_weights.get()->info()->strides_in_bytes()[width_idx] / element_size_in_bytes;
        const int permuted_weight_channel_stride = permuted_weights.get()->info()->strides_in_bytes()[channel_idx] / element_size_in_bytes;

        // Wrap the winograd-domain transformed weight TensorInfo in Auxiliary tensor and allocate the required memory.
        ITensor *weights_transf = utils::cast::polymorphic_cast<ITensor *>(tensors.get_tensor(offset_int_vec(TransformedWeights)));
        ARM_COMPUTE_ERROR_ON_NULLPTR(weights_transf);
        CpuAuxTensorHandler winograd_transformed_weights(_winograd_transformed_weights, *weights_transf);

        const void *permuted_weights_ptr;
        void       *win_wght_transf_ptr;

        permuted_weights_ptr = reinterpret_cast<const void *>(permuted_weights.get()->buffer() + permuted_weights.get()->info()->offset_first_element_in_bytes());
        win_wght_transf_ptr  = reinterpret_cast<void *>(winograd_transformed_weights.get()->buffer() + winograd_transformed_weights.get()->info()->offset_first_element_in_bytes());

        // Prepare Weights
        _winograd_impl.weight_transform->execute(
            *_conv_args,
            permuted_weights_ptr,
            permuted_weight_row_stride,
            permuted_weight_col_stride,
            permuted_weight_channel_stride,
            win_wght_transf_ptr,
            _winograd_impl.winograd_spec,
            0, 1 // Thread 1 of 1
        );
        ITensorPack gemm_pack = tensors;
        gemm_pack.add_const_tensor(ACL_SRC_1, winograd_transformed_weights.get());
        _gemm_function->prepare(gemm_pack);
        _is_prepared = 1;
    }
}
experimental::MemoryRequirements CpuWinogradConv2d::workspace() const
{
    return _aux_mem;
}

} // namespace cpu
} // namespace arm_compute
