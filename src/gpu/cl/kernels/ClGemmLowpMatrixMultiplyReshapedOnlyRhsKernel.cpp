/*
 * Copyright (c) 2019-2022 Arm Limited.
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
#include "src/gpu/cl/kernels/ClGemmLowpMatrixMultiplyReshapedOnlyRhsKernel.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include "src/core/AccessWindowStatic.h"
#include "src/core/helpers/AutoConfiguration.h"
#include "src/core/helpers/WindowHelpers.h"

#include "support/Cast.h"
#include "support/StringSupport.h"

#include <tuple>

namespace arm_compute
{
namespace opencl
{
namespace kernels
{
using namespace misc::shape_calculator;

namespace
{
using ElementsProcessed = Steps;

Status validate_arguments(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *dst, const GEMMKernelInfo &gemm_info,
                          const ITensorInfo *vector_sum_col, const ITensorInfo *vector_sum_row, const ITensorInfo *bias,
                          const ITensorInfo *output_multipliers, const ITensorInfo *output_shifts)
{
    ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(src0, src1, dst);
    ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src0, 1, DataType::QASYMM8, DataType::QASYMM8_SIGNED);
    if(src0->data_type() == DataType::QASYMM8)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src0, src1);
    }
    else
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(src1, 1, DataType::QASYMM8, DataType::QSYMM8, DataType::QASYMM8_SIGNED, DataType::QSYMM8_PER_CHANNEL);
    }
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(src0->num_dimensions() > 4, "The number of dimensions for the LHS matrix must be <= 4");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(src1->num_dimensions() > 3, "The number of dimensions for the RHS matrix must be <= 3");

    const GEMMRHSMatrixInfo       rhs_info     = gemm_info.rhs_info;
    const GEMMLHSMatrixInfo       lhs_info     = gemm_info.lhs_info;
    const GEMMLowpOutputStageInfo output_stage = gemm_info.output_stage;

    ARM_COMPUTE_RETURN_ERROR_ON_MSG((((rhs_info.k0 & (rhs_info.k0 - 1)) && rhs_info.k0 != 3) || (rhs_info.k0 > 16)), "Only 2,3,4,8,16 are supported for k0");
    ARM_COMPUTE_RETURN_ERROR_ON(lhs_info.m0 < 1 || lhs_info.m0 > 8);
    ARM_COMPUTE_RETURN_ERROR_ON_MSG((((rhs_info.n0 & (rhs_info.n0 - 1)) && rhs_info.n0 != 3) || rhs_info.n0 > 16), "Only 2,3,4,8,16 are supported for n0");
    ARM_COMPUTE_RETURN_ERROR_ON_MSG(rhs_info.export_to_cl_image, "Export to CLImage not supported for quantized GEMM");

    const int m = gemm_info.m;
    const int n = gemm_info.n;
    const int k = gemm_info.k;

    TensorShape tensor_shape1{ src1->tensor_shape() };
    tensor_shape1.set(0, n);
    tensor_shape1.set(1, k);

    const TensorInfo tensor_info1          = src1->clone()->set_tensor_shape(tensor_shape1);
    const TensorInfo tensor_info_reshaped1 = src1->clone()->set_tensor_shape(compute_rhs_reshaped_shape(tensor_info1, rhs_info));

    ARM_COMPUTE_RETURN_ERROR_ON(src0->dimension(0) != static_cast<unsigned int>(k));
    if(gemm_info.reinterpret_input_as_3d)
    {
        ARM_COMPUTE_RETURN_ERROR_ON(src0->dimension(1) * src0->dimension(2) != static_cast<unsigned int>(m));
    }
    else
    {
        ARM_COMPUTE_RETURN_ERROR_ON(src0->dimension(1) != static_cast<unsigned int>(m));
    }
    ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(src1, &tensor_info_reshaped1);

    const TensorShape expected_dst_shape = compute_mm_shape(*src0, *src1, gemm_info);
    if(dst->total_size() != 0)
    {
        const TensorInfo tensor_info_dst = dst->clone()->set_tensor_shape(expected_dst_shape);
        ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_SHAPES(dst, &tensor_info_dst);
        if(output_stage.type == GEMMLowpOutputStageType::NONE)
        {
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(dst, 1, DataType::S32);
        }
        else
        {
            ARM_COMPUTE_RETURN_ERROR_ON_MISMATCHING_DATA_TYPES(src0, dst);
        }
    }

    if(bias != nullptr)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(bias, 1, DataType::S32);
        ARM_COMPUTE_RETURN_ERROR_ON(bias->num_dimensions() > 1);
        ARM_COMPUTE_RETURN_ERROR_ON(expected_dst_shape[0] != bias->dimension(0));
    }

    ARM_COMPUTE_RETURN_ERROR_ON_MSG((output_stage.type == GEMMLowpOutputStageType::QUANTIZE_DOWN) || (output_stage.type == GEMMLowpOutputStageType::QUANTIZE_DOWN_FLOAT),
                                    "Only GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT is supported");

    // Checks performed if the dst stage needs to be fused
    if(output_stage.type == GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT)
    {
        // If a_offset == 0, vector_sum_col can be a nullptr
        if(gemm_info.a_offset != 0)
        {
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(vector_sum_col, 1, DataType::S32);
            ARM_COMPUTE_RETURN_ERROR_ON(vector_sum_col->dimension(0) != expected_dst_shape[0]);
        }

        // If b_offset == 0, vector_sum_row can be a nullptr
        if(gemm_info.b_offset != 0)
        {
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(vector_sum_row, 1, DataType::S32);

            // Check if mm result is a 3D reinterpretation
            const bool reinterpret_as_3d = expected_dst_shape.num_dimensions() > 1 && expected_dst_shape.y() != vector_sum_row->tensor_shape().x();

            // Validate input
            ARM_COMPUTE_RETURN_ERROR_ON(reinterpret_as_3d && vector_sum_row->dimension(0) != (expected_dst_shape[1] * expected_dst_shape[2]));
            ARM_COMPUTE_RETURN_ERROR_ON(!reinterpret_as_3d && vector_sum_row->dimension(0) != expected_dst_shape[1]);

            if(expected_dst_shape.num_dimensions() > 1)
            {
                const unsigned int dst_batch_idx = reinterpret_as_3d ? 3 : 2;

                TensorShape vector_sum_row_shape = vector_sum_row->tensor_shape();
                vector_sum_row_shape.collapse_from(1);
                TensorShape collapsed_dst_shape(expected_dst_shape);
                collapsed_dst_shape.collapse_from(dst_batch_idx);

                ARM_COMPUTE_RETURN_ERROR_ON_MSG(vector_sum_row_shape[1] != collapsed_dst_shape[dst_batch_idx],
                                                "vector_sum_row must have the same number of batches of dst tensor");

                if(gemm_info.a_offset != 0)
                {
                    TensorShape vector_sum_col_shape = vector_sum_col->tensor_shape();
                    vector_sum_col_shape.collapse_from(1);

                    ARM_COMPUTE_RETURN_ERROR_ON_MSG(vector_sum_col_shape[1] != 1 && vector_sum_col_shape[1] != vector_sum_row_shape[1],
                                                    "vector_sum_col tensor must have the same number of batches of vector_sum_row_shape or the number of batches must be set to 1");
                }
            }
        }

        if(dst->total_size() != 0)
        {
            ARM_COMPUTE_RETURN_ERROR_ON(output_stage.output_data_type != dst->data_type());
        }
        ARM_COMPUTE_RETURN_ERROR_ON(output_stage.gemmlowp_min_bound > output_stage.gemmlowp_max_bound);

        if(output_multipliers != nullptr && output_shifts != nullptr)
        {
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output_multipliers, 1, DataType::S32);
            ARM_COMPUTE_RETURN_ERROR_ON(output_multipliers->num_dimensions() > 1);
            ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(output_shifts, 1, DataType::S32);
            ARM_COMPUTE_RETURN_ERROR_ON(output_shifts->num_dimensions() > 1);
            if(output_stage.is_quantized_per_channel)
            {
                ARM_COMPUTE_RETURN_ERROR_ON(expected_dst_shape[0] != output_shifts->dimension(0));
                ARM_COMPUTE_RETURN_ERROR_ON(expected_dst_shape[0] != output_multipliers->dimension(0));
            }
        }
    }
    return Status{};
}

std::pair<Status, Window> validate_and_configure_window(const ITensorInfo *src0, const ITensorInfo *src1, ITensorInfo *dst, const GEMMKernelInfo &gemm_info,
                                                        ITensorInfo *vector_sum_col, const ITensorInfo *vector_sum_row, ITensorInfo *bias,
                                                        ITensorInfo *output_multipliers, ITensorInfo *output_shifts, ElementsProcessed &num_elements_processed)
{
    const GEMMLowpOutputStageInfo output_stage = gemm_info.output_stage;

    unsigned int &num_elems_processed_per_iteration_x = num_elements_processed[0];
    unsigned int &num_elems_processed_per_iteration_y = num_elements_processed[1];
    bool          reinterpret_input_as_3d             = gemm_info.reinterpret_input_as_3d;
    bool          reinterpret_output_as_3d            = (gemm_info.depth_output_gemm3d != 0);

    Window win{};
    Window win_out{};
    bool   window_changed = false;

    // In case both input and dst have to be reinterpreted as 3D tensors,
    // force reinterpret_input_as_3d and reinterpret_output_as_3d to be false.
    if(reinterpret_input_as_3d == reinterpret_output_as_3d)
    {
        reinterpret_output_as_3d = false;
    }

    // dst tensor auto initialization if not yet initialized
    const TensorShape expected_dst_shape = compute_mm_shape(*src0, *src1, gemm_info);
    if(output_stage.type != GEMMLowpOutputStageType::NONE)
    {
        auto_init_if_empty(*dst, src0->clone()->set_tensor_shape(expected_dst_shape).set_data_type(output_stage.output_data_type));
    }
    else
    {
        auto_init_if_empty(*dst, src0->clone()->set_tensor_shape(expected_dst_shape).set_data_type(DataType::S32));
    }

    TensorInfo tmp_info(*dst);

    if(reinterpret_output_as_3d)
    {
        // Since the dst tensor has to be reinterpreted as 3D and the execute window is based on a 2D GEMM,
        // the window needs to be constructed on the 2D collapsed version of the tensor
        TensorShape tmp_shape(dst->tensor_shape());
        tmp_shape.collapse(2U, 1U);
        tmp_info.set_tensor_shape(tmp_shape);
    }

    // Configure kernel window
    num_elems_processed_per_iteration_x = gemm_info.rhs_info.n0;
    num_elems_processed_per_iteration_y = gemm_info.lhs_info.m0;

    win     = calculate_max_window(tmp_info, Steps(num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y));
    win_out = calculate_max_window(*dst, Steps(num_elems_processed_per_iteration_x, num_elems_processed_per_iteration_y));

    if(output_stage.type == GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT)
    {
        if(gemm_info.a_offset != 0)
        {
            AccessWindowHorizontal vector_sum_col_access(vector_sum_col, 0, num_elems_processed_per_iteration_x);
            window_changed = window_changed || update_window_and_padding(win_out, vector_sum_col_access);
        }
        // No access window needed for vector_sum_row
        ARM_COMPUTE_UNUSED(vector_sum_row);

        if(bias != nullptr)
        {
            AccessWindowHorizontal bias_access(bias, 0, num_elems_processed_per_iteration_x);
            window_changed = window_changed || update_window_and_padding(win_out, bias_access);
        }

        if(output_multipliers != nullptr && output_stage.is_quantized_per_channel)
        {
            AccessWindowHorizontal output_multipliers_access(output_multipliers, 0, num_elems_processed_per_iteration_x);
            AccessWindowHorizontal output_shifts_access(output_shifts, 0, num_elems_processed_per_iteration_x);
            window_changed = window_changed || update_window_and_padding(win_out, output_multipliers_access, output_shifts_access);
        }
    }

    // Collapse along the Z direction
    // This collapse needs to be here in order to tune the Z dimension of LWS
    Window             collapsed             = win;
    const unsigned int dimension_to_collapse = std::min(static_cast<unsigned int>(dst->num_dimensions()), 2u);
    collapsed                                = win.collapse(win, dimension_to_collapse);

    Status err = (window_changed) ? ARM_COMPUTE_CREATE_ERROR(ErrorCode::RUNTIME_ERROR, "Insufficient Padding!") : Status{};
    return std::make_pair(err, collapsed);
}
} // namespace

ClGemmLowpMatrixMultiplyReshapedOnlyRhsKernel::ClGemmLowpMatrixMultiplyReshapedOnlyRhsKernel()
{
    _type = CLKernelType::GEMM;
}

void ClGemmLowpMatrixMultiplyReshapedOnlyRhsKernel::configure(const CLCompileContext &compile_context, const ITensorInfo *src0, const ITensorInfo *src1, ITensorInfo *dst,
                                                              const GEMMKernelInfo &gemm_info,
                                                              ITensorInfo *vector_sum_col, const ITensorInfo *vector_sum_row, ITensorInfo *bias,
                                                              ITensorInfo *output_multipliers, ITensorInfo *output_shifts)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(src0, src1, dst);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(src0, src1, dst, gemm_info, vector_sum_col, vector_sum_row, bias, output_multipliers, output_shifts));

    auto                          padding_info = get_padding_info({ src0, src1, dst, vector_sum_row });
    const GEMMRHSMatrixInfo       rhs_info     = gemm_info.rhs_info;
    const GEMMLHSMatrixInfo       lhs_info     = gemm_info.lhs_info;
    const GEMMLowpOutputStageInfo output_stage = gemm_info.output_stage;
    const int32_t                 a_offset     = gemm_info.a_offset;
    const int32_t                 b_offset     = gemm_info.b_offset;

    _reinterpret_input_as_3d  = gemm_info.reinterpret_input_as_3d;
    _reinterpret_output_as_3d = (gemm_info.depth_output_gemm3d != 0);
    _use_dummy_work_items     = preferred_dummy_work_items_support(CLKernelLibrary::get().get_device());
    _is_quantized_per_channel = output_stage.is_quantized_per_channel;

    // In case both input and dst have to be reinterpreted as 3D tensors,
    // force reinterpret_input_as_3d and reinterpret_output_as_3d to be false.
    if(_reinterpret_input_as_3d == _reinterpret_output_as_3d)
    {
        _reinterpret_input_as_3d  = false;
        _reinterpret_output_as_3d = false;
    }

    // Check if we need to slide the matrix B
    const unsigned int num_dimensions_src0 = src0->num_dimensions();
    _slide_matrix_b                        = (src1->num_dimensions() >= num_dimensions_src0);

    ElementsProcessed num_elements_processed{};

    // Configure kernel window
    auto win_config = validate_and_configure_window(src0, src1, dst, gemm_info, vector_sum_col, vector_sum_row, bias, output_multipliers, output_shifts, num_elements_processed);
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    ICLKernel::configure_internal(win_config.second);

    // If _reinterpret_input_as_3d = _reinterpret_output_as_3d = true,
    // we will dispatch a batched-GEMM to reduce the complexity of the address calculation within the OpenCL kernel.
    // This means that the actual m used by the kernel is given by dst->dimension(1) and not by gemm_info.m
    const unsigned int internal_m = _reinterpret_output_as_3d ? gemm_info.m : dst->dimension(1);

    // Shrink M0 to be always <= M (internal_m) to prevent out-of-bounds reads.
    // NOTE: This might have implications on heuristics and performance
    const unsigned int internal_m0 = std::min(internal_m, lhs_info.m0);

    // Calculate partial (store instead of load) M0 and partial N0 for the partial blocks at the end of a row/column if any. This is to avoid padding.
    const unsigned int partial_store_m0 = internal_m % internal_m0;
    const unsigned int partial_store_n0 = gemm_info.n % rhs_info.n0;

    // Create build options
    CLBuildOptions build_opts;
    build_opts.add_option_if(_reinterpret_input_as_3d, "-DREINTERPRET_INPUT_AS_3D");
    build_opts.add_option_if(_reinterpret_output_as_3d, "-DREINTERPRET_OUTPUT_AS_3D");
    build_opts.add_option_if(_reinterpret_input_as_3d || _reinterpret_output_as_3d, "-DHEIGHT_GEMM3D=" + support::cpp11::to_string(dst->dimension(1)));
    build_opts.add_option_if(_reinterpret_input_as_3d || _reinterpret_output_as_3d, "-DDEPTH_GEMM3D=" + support::cpp11::to_string(dst->dimension(2)));
    build_opts.add_option_if(!_slide_matrix_b, "-DMATRIX_B_DEPTH=" + support::cpp11::to_string(src1->dimension(2)));
    build_opts.add_option_if(rhs_info.interleave, "-DRHS_INTERLEAVE");
    build_opts.add_option_if(_use_dummy_work_items, "-DDUMMY_WORK_ITEMS");
    build_opts.add_option("-DM=" + support::cpp11::to_string(internal_m));
    build_opts.add_option("-DN=" + support::cpp11::to_string(gemm_info.n));
    build_opts.add_option("-DK=" + support::cpp11::to_string(gemm_info.k));
    build_opts.add_option("-DM0=" + support::cpp11::to_string(internal_m0));
    build_opts.add_option("-DN0=" + support::cpp11::to_string(rhs_info.n0));
    build_opts.add_option("-DK0=" + support::cpp11::to_string(rhs_info.k0));
    build_opts.add_option("-DH0=" + support::cpp11::to_string(rhs_info.h0));
    build_opts.add_option("-DPARTIAL_STORE_M0=" + support::cpp11::to_string(partial_store_m0));
    build_opts.add_option("-DPARTIAL_STORE_N0=" + support::cpp11::to_string(partial_store_n0));
    build_opts.add_option("-DDATA_TYPE=" + get_cl_type_from_data_type(src0->data_type()));
    build_opts.add_option("-DACC_DATA_TYPE=" + get_cl_dot8_acc_type_from_data_type(src0->data_type()));

    std::string kernel_name("gemmlowp_mm_reshaped_only_rhs_");
    kernel_name += rhs_info.transpose ? "t" : "nt";

    if(output_stage.type == GEMMLowpOutputStageType::QUANTIZE_DOWN_FIXEDPOINT)
    {
        kernel_name += "_fused_output_stage_fixedpoint";
        _fuse_output_stage = true;
        // If a_offset == 0, vector_sum_col can be a nullptr
        if(a_offset != 0 && vector_sum_col != nullptr)
        {
            build_opts.add_option("-DA_OFFSET=" + support::cpp11::to_string(a_offset));
            build_opts.add_option_if(vector_sum_col->tensor_shape().num_dimensions() > 1, "-DSUM_COL_HAS_BATCHES");
        }
        // If b_offset == 0, vector_sum_row can be a nullptr
        build_opts.add_option_if(b_offset != 0, "-DB_OFFSET=" + support::cpp11::to_string(b_offset));
        build_opts.add_option("-DK_OFFSET=" + support::cpp11::to_string(a_offset * b_offset * src0->dimension(0)));
        build_opts.add_option_if(bias != nullptr, "-DADD_BIAS");
        build_opts.add_option("-DRESULT_OFFSET=" + support::cpp11::to_string(output_stage.gemmlowp_offset));
        build_opts.add_option("-DRESULT_MULTIPLIER=" + support::cpp11::to_string(output_stage.gemmlowp_multipliers[0]));
        build_opts.add_option("-DRESULT_SHIFT=" + support::cpp11::to_string(output_stage.gemmlowp_shifts[0]));
        build_opts.add_option_if(_is_quantized_per_channel, "-DPER_CHANNEL_QUANTIZATION");

        const int min = output_stage.gemmlowp_min_bound;
        const int max = output_stage.gemmlowp_max_bound;

        PixelValue min_val{};
        PixelValue max_val{};
        std::tie(min_val, max_val) = get_min_max(dst->data_type());
        build_opts.add_option_if(min != min_val.get<int32_t>(), "-DMIN_BOUND=" + support::cpp11::to_string(min));
        build_opts.add_option_if(max != max_val.get<int32_t>(), "-DMAX_BOUND=" + support::cpp11::to_string(max));
    }

    // A macro guard to compile ONLY the kernel of interest
    build_opts.add_option("-D" + upper_string(kernel_name));

    // Create kernel
    _kernel = create_kernel(compile_context, kernel_name, build_opts.options());

    // Set config_id for enabling LWS tuning
    _config_id = kernel_name;
    _config_id += "_";
    _config_id += dot8_supported(CLKernelLibrary::get().get_device()) ? "_dot8" : "";
    _config_id += "_";
    _config_id += (_reinterpret_input_as_3d ? "3di_" : "");
    _config_id += (_reinterpret_output_as_3d ? "3do_" : "");
    _config_id += support::cpp11::to_string(dst->dimension(1));
    _config_id += "_";
    _config_id += support::cpp11::to_string(dst->dimension(0));
    _config_id += "_";
    _config_id += support::cpp11::to_string(gemm_info.k);
    _config_id += "_";
    _config_id += support::cpp11::to_string(dst->dimension(2));
    _config_id += "_";
    _config_id += support::cpp11::to_string(lhs_info.m0);
    _config_id += "_";
    _config_id += support::cpp11::to_string(rhs_info.n0);
    _config_id += "_";
    _config_id += support::cpp11::to_string(rhs_info.k0);
    _config_id += "_";
    _config_id += support::cpp11::to_string(rhs_info.h0);
    _config_id += "_";
    _config_id += support::cpp11::to_string(rhs_info.interleave);
    ARM_COMPUTE_ERROR_ON(has_padding_changed(padding_info));
}

Status ClGemmLowpMatrixMultiplyReshapedOnlyRhsKernel::validate(const ITensorInfo *src0, const ITensorInfo *src1, const ITensorInfo *dst, const GEMMKernelInfo &gemm_info,
                                                               const ITensorInfo *vector_sum_col, const ITensorInfo *vector_sum_row, const ITensorInfo *bias,
                                                               const ITensorInfo *output_multipliers, const ITensorInfo *output_shifts)
{
    ElementsProcessed num_elements_processed{};
    ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(src0, src1, dst, gemm_info, vector_sum_col, vector_sum_row, bias, output_multipliers, output_shifts));
    ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(src0->clone().get(),
                                                              src1->clone().get(),
                                                              dst->clone().get(),
                                                              gemm_info,
                                                              vector_sum_col != nullptr ? vector_sum_col->clone().get() : nullptr,
                                                              vector_sum_row != nullptr ? vector_sum_row->clone().get() : nullptr,
                                                              bias != nullptr ? bias->clone().get() : nullptr,
                                                              output_multipliers != nullptr ? output_multipliers->clone().get() : nullptr,
                                                              output_shifts != nullptr ? output_shifts->clone().get() : nullptr,
                                                              num_elements_processed)
                                .first);

    return Status{};
}

void ClGemmLowpMatrixMultiplyReshapedOnlyRhsKernel::run_op(ITensorPack &tensors, const Window &window, cl::CommandQueue &queue)
{
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(ICLKernel::window(), window);

    const auto src0               = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_0));
    const auto src1               = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SRC_1));
    const auto bias               = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_BIAS));
    const auto vector_sum_col     = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_VEC_COL_SUM));
    const auto vector_sum_row     = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_VEC_ROW_SUM));
    const auto output_shifts      = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_SHIFTS));
    const auto output_multipliers = utils::cast::polymorphic_downcast<const ICLTensor *>(tensors.get_const_tensor(TensorType::ACL_MULTIPLIERS));
    auto       dst                = utils::cast::polymorphic_downcast<ICLTensor *>(tensors.get_tensor(TensorType::ACL_DST));

    if(src1->info()->num_dimensions() < 3)
    {
        // The stride_z for matrix B must be zero if we do not slice
        ARM_COMPUTE_ERROR_ON(src1->info()->strides_in_bytes()[3] != 0);
    }

    Window slice          = window.first_slice_window_3D();
    Window slice_matrix_b = slice;

    slice_matrix_b.set(Window::DimX, Window::Dimension(0, 1, 1));
    slice_matrix_b.set(Window::DimY, Window::Dimension(0, 1, 1));

    if(_reinterpret_input_as_3d)
    {
        // Pass bottom paddings to the kernel if the input has to be reinterpreted as 3D tensor
        const unsigned int idx0                  = 3 * num_arguments_per_2D_tensor() + 3;
        const unsigned int total_cross_plane_pad = src0->info()->padding().top + src0->info()->padding().bottom;
        _kernel.setArg<cl_uint>(idx0, static_cast<unsigned int>(total_cross_plane_pad));
    }

    if(_reinterpret_output_as_3d)
    {
        // Pass bottom paddings to the kernel if the dst has to be reinterpreted as 3D tensor
        const unsigned int idx0                  = 3 * num_arguments_per_2D_tensor() + 3 + (_reinterpret_input_as_3d ? 1 : 0);
        const unsigned int total_cross_plane_pad = dst->info()->padding().top + dst->info()->padding().bottom;
        _kernel.setArg<cl_uint>(idx0, static_cast<unsigned int>(total_cross_plane_pad));
    }

    // Set window for vector_sum_col
    Window win_vector_sum_col = slice;
    win_vector_sum_col.set(Window::DimY, Window::Dimension(0, 0, 0));
    win_vector_sum_col.set(Window::DimZ, Window::Dimension(0, 0, 0));

    // Set window for vector_sum_row
    Window win_vector_sum_row = slice;
    win_vector_sum_row.set(Window::DimX, Window::Dimension(0, 0, 0));
    win_vector_sum_row.set(Window::DimY, Window::Dimension(0, 0, 0));
    win_vector_sum_col.set(Window::DimZ, Window::Dimension(0, 0, 0));

    Window biases_slice = slice;
    biases_slice.set(Window::DimY, Window::Dimension(0, 1, 1));
    biases_slice.set(Window::DimZ, Window::Dimension(0, 1, 1));

    do
    {
        Window slice_b = slice;
        // Don't slice matrix B along the z dimension if matrix B has just 2 dimensions and matrix A more than 2
        // This scenario can happen when the matrix multiplication is used to perform a convolution operation
        if(!_slide_matrix_b)
        {
            slice_b = slice_matrix_b;
        }

        unsigned int idx = 0;
        add_2D_tensor_argument(idx, src0, slice);
        add_2D_tensor_argument(idx, src1, slice_b);
        add_2D_tensor_argument(idx, dst, slice);
        _kernel.setArg<cl_uint>(idx++, static_cast<unsigned int>(src0->info()->strides_in_bytes()[2]));
        _kernel.setArg<cl_uint>(idx++, static_cast<unsigned int>(src1->info()->strides_in_bytes()[2]));
        _kernel.setArg<cl_uint>(idx++, static_cast<unsigned int>(dst->info()->strides_in_bytes()[2]));
        if(_reinterpret_input_as_3d)
        {
            // Pass bottom paddings to the kernel if the input has to be reinterpreted as 3D tensor
            idx++;
        }

        if(_reinterpret_output_as_3d)
        {
            // Pass bottom paddings to the kernel if the dst has to be reinterpreted as 3D tensor
            idx++;
        }

        if(_fuse_output_stage)
        {
            add_2D_tensor_argument_if((vector_sum_col != nullptr), idx, vector_sum_col, win_vector_sum_col);
            add_2D_tensor_argument_if((vector_sum_row != nullptr), idx, vector_sum_row, win_vector_sum_row);
            add_1D_tensor_argument_if((bias != nullptr), idx, bias, biases_slice);
            add_1D_tensor_argument_if(_is_quantized_per_channel, idx, output_multipliers, biases_slice);
            add_1D_tensor_argument_if(_is_quantized_per_channel, idx, output_shifts, biases_slice);
        }
        enqueue(queue, *this, slice, lws_hint(), _use_dummy_work_items);
    }
    while(window.slide_window_slice_3D(slice));
}
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
