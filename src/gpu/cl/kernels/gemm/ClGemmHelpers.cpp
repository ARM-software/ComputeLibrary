/*
 * Copyright (c) 2019-2023 Arm Limited.
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
#include "src/gpu/cl/kernels/gemm/ClGemmHelpers.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include <limits>
#include <utility>

namespace arm_compute
{
namespace opencl
{
namespace kernels
{
namespace gemm
{
std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> configure_lhs_rhs_info(unsigned int m, unsigned int n, unsigned int m0, unsigned int n0, unsigned int k0, unsigned int v0, unsigned int h0,
                                                                       bool lhs_interleave, bool rhs_interleave, bool lhs_transpose, bool rhs_transpose, bool export_to_cl_image)
{
    ARM_COMPUTE_ERROR_ON(m0 == 0 || n0 == 0);
    ARM_COMPUTE_ERROR_ON(v0 == 0);
    v0 = std::max(std::min(static_cast<int>(m / m0), static_cast<int>(v0)), static_cast<int>(1));

    if(h0 == 0)
    {
        // When h0 is 0, we should take the maximum H0 possible
        h0 = std::max(n / n0, 1U);
    }
    else
    {
        h0 = std::max(std::min(static_cast<int>(n / n0), static_cast<int>(h0)), static_cast<int>(1));
    }

    const GEMMLHSMatrixInfo lhs_info(m0, k0, v0, lhs_transpose, lhs_interleave);
    const GEMMRHSMatrixInfo rhs_info(n0, k0, h0, rhs_transpose, rhs_interleave, export_to_cl_image);

    return std::make_pair(lhs_info, rhs_info);
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> select_lhs_rhs_info(std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> info_img,
                                                                    std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> info_buf,
                                                                    unsigned int n, unsigned int k, unsigned int b, DataType data_type)
{
    ARM_COMPUTE_ERROR_ON_MSG(info_buf.second.export_to_cl_image == true, "The fallback GeMM configuration cannot have export_to_cl_image = true");

    const TensorInfo  tensor_rhs_info(TensorShape(n, k, b), 1, data_type);
    const TensorShape shape = misc::shape_calculator::compute_rhs_reshaped_shape(tensor_rhs_info, info_img.second);
    const TensorInfo  tensor_reshaped_info(shape, 1, data_type);

    if(bool(validate_image2d_support_on_rhs(tensor_reshaped_info, info_img.second)))
    {
        return info_img;
    }
    else
    {
        return info_buf;
    }
}

void update_padding_for_cl_image(ITensorInfo *tensor)
{
    constexpr unsigned int num_floats_per_pixel = 4;

    const unsigned int stride_y_in_elements = tensor->strides_in_bytes()[1] / tensor->element_size();
    const unsigned int pixel_alignment      = get_cl_image_pitch_alignment(CLKernelLibrary::get().get_device());

    ARM_COMPUTE_ERROR_ON_MSG(pixel_alignment == 0, "Cannot retrieve cl_image pitch alignment");
    if(pixel_alignment == 0)
    {
        return;
    }

    const unsigned int row_pitch_alignment = pixel_alignment * num_floats_per_pixel;
    const unsigned int round_up_width      = ((stride_y_in_elements + row_pitch_alignment - 1) / row_pitch_alignment) * row_pitch_alignment;
    const unsigned int padding             = round_up_width - stride_y_in_elements;

    tensor->extend_padding(PaddingSize(0, tensor->padding().right + padding, 0, 0));
}

Status validate_image2d_support_on_rhs(const ITensorInfo &tensor_reshaped_info, const GEMMRHSMatrixInfo &rhs_info)
{
    if(rhs_info.export_to_cl_image)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(((rhs_info.n0 == 2) || (rhs_info.n0 == 3)) && rhs_info.transpose == false, "Export to cl_image only supported with n0 = 4, 8 or 16");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(((rhs_info.k0 == 2) || (rhs_info.k0 == 3)) && rhs_info.transpose == true, "Export to cl_image only supported with k0 = 4, 8 or 16");
        ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_NOT_IN(&tensor_reshaped_info, DataType::F32, DataType::F16);
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(!image2d_from_buffer_supported(CLKernelLibrary::get().get_device()), "The extension cl_khr_image2d_from_buffer is not supported on the target platform");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(get_cl_image_pitch_alignment(CLKernelLibrary::get().get_device()) == 0, "Impossible to retrieve the cl_image pitch alignment");

        // Check the width and height of the output tensor.
        // Since we cannot create a 3d image from a buffer, the third dimension is collapsed on the second dimension
        const size_t max_image_w = CLKernelLibrary::get().get_device().getInfo<CL_DEVICE_IMAGE2D_MAX_WIDTH>();
        const size_t max_image_h = CLKernelLibrary::get().get_device().getInfo<CL_DEVICE_IMAGE2D_MAX_HEIGHT>();

        ARM_COMPUTE_RETURN_ERROR_ON_MSG(tensor_reshaped_info.tensor_shape()[0] > max_image_w * 4, "Not supported width for cl_image");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(tensor_reshaped_info.tensor_shape()[1] * tensor_reshaped_info.tensor_shape()[2] > max_image_h, "Not supported height for cl_image");
    }

    return Status{};
}

bool is_mmul_kernel_preferred(const unsigned int m, const unsigned int n, const unsigned int k, const unsigned int b,
                              const DataType data_type, unsigned int &best_m0, unsigned int &best_n0)
{
    ARM_COMPUTE_UNUSED(n, k, b, data_type);

    const unsigned int mmul_k0 = 4;
    best_m0                    = 4;
    best_n0                    = 4;

    const unsigned int ceil_to_multiple_m_m0             = ceil_to_multiple(m, best_m0);
    const unsigned int m_div_m0                          = ceil_to_multiple_m_m0 / best_m0;
    const unsigned int ceil_to_multiple_m_div_m0_mmul_k0 = ceil_to_multiple(m_div_m0, mmul_k0);
    const unsigned int gws_y                             = ceil_to_multiple_m_div_m0_mmul_k0 / mmul_k0;

    return ((k % mmul_k0) == 0) && (gws_y > 4);
}

std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> find_lhs_rhs_info(const GeMMConfigsMatrix &configs, unsigned int m, unsigned int n, unsigned int k, unsigned int b)
{
    float  min_acc = std::numeric_limits<float>::max();
    size_t min_idx = 0;

    ARM_COMPUTE_ERROR_ON(configs.size() == 0);
    const size_t num_rows = configs.size();
    const size_t num_cols = configs[0].size();

    ARM_COMPUTE_ERROR_ON_MSG(num_cols != 14U, "The entry should have 14 integer values representing: M, N, K, B, M0, N0. K0, V0, H0, INT_LHS, INT_RHS, TRA_LHS, TRA_RHS, IMG_RHS");
    ARM_COMPUTE_UNUSED(num_cols);

    // Find nearest GeMM shape
    for(size_t y = 0; y < num_rows; ++y)
    {
        float mc0 = configs[y][0];
        float nc0 = configs[y][1];
        float kc0 = configs[y][2];
        float bc0 = configs[y][3];
        float acc = 0;
        acc += (m - mc0) * (m - mc0);
        acc += (n - nc0) * (n - nc0);
        acc += (k - kc0) * (n - kc0);
        acc += (b - bc0) * (n - bc0);
        acc = std::sqrt(acc);
        if(acc < min_acc)
        {
            min_acc = acc;
            min_idx = y;
        }
    }

    // Get the configuration from the nearest GeMM shape
    const int m0     = configs[min_idx][4];
    const int n0     = configs[min_idx][5];
    const int k0     = configs[min_idx][6];
    const int v0     = configs[min_idx][7];
    const int h0     = configs[min_idx][8];
    const int i_lhs  = configs[min_idx][9];
    const int i_rhs  = configs[min_idx][10];
    const int t_lhs  = configs[min_idx][11];
    const int t_rhs  = configs[min_idx][12];
    const int im_rhs = configs[min_idx][13];

    return configure_lhs_rhs_info(m, n, m0, n0, k0, v0, h0, i_lhs, i_rhs, t_lhs, t_rhs, im_rhs);
}
} // namespace gemm
} // namespace kernels
} // namespace opencl
} // namespace arm_compute
