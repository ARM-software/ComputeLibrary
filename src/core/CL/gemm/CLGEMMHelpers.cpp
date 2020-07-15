/*
 * Copyright (c) 2019-2020 Arm Limited.
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
#include "arm_compute/core/CL/gemm/CLGEMMHelpers.h"

#include "arm_compute/core/CL/CLHelpers.h"
#include "arm_compute/core/CL/CLKernelLibrary.h"
#include "arm_compute/core/CL/OpenCL.h"
#include "arm_compute/core/ITensorInfo.h"

#include <utility>

namespace arm_compute
{
namespace cl_gemm
{
std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> configure_lhs_rhs_info(unsigned int m, unsigned int n, unsigned int m0, unsigned int n0, unsigned int k0, unsigned int v0, unsigned int h0,
                                                                       bool lhs_interleave, bool rhs_interleave, bool lhs_transpose, bool rhs_transpose, bool export_to_cl_image)
{
    v0 = ((m / (m0 * v0)) == 0) ? 1 : v0;
    h0 = ((n / (n0 * h0)) == 0) ? 1 : h0;

    const GEMMLHSMatrixInfo lhs_info(m0, k0, v0, lhs_transpose, lhs_interleave);
    const GEMMRHSMatrixInfo rhs_info(n0, k0, h0, rhs_transpose, rhs_interleave, export_to_cl_image);

    return std::make_pair(lhs_info, rhs_info);
}

void update_padding_for_cl_image(ITensorInfo *tensor)
{
    constexpr unsigned int num_floats_per_pixel = 4;

    const unsigned int stride_y_in_elements = tensor->strides_in_bytes()[1] / tensor->element_size();
    const unsigned int pixel_aligment       = get_cl_image_pitch_alignment(CLKernelLibrary::get().get_device());
    const unsigned int row_pitch_alignment  = pixel_aligment * num_floats_per_pixel;
    const unsigned int round_up_width       = ((stride_y_in_elements + row_pitch_alignment - 1) / row_pitch_alignment) * row_pitch_alignment;
    const unsigned int padding              = round_up_width - stride_y_in_elements;

    tensor->extend_padding(PaddingSize(0, padding, 0, 0));
}

Status validate_image2d_support_on_rhs(const ITensorInfo &tensor_reshaped_info, const GEMMRHSMatrixInfo &rhs_info)
{
    if(rhs_info.export_to_cl_image)
    {
        ARM_COMPUTE_RETURN_ERROR_ON_MSG((rhs_info.n0 == 2) || (rhs_info.n0 == 3), "Export to cl_image only supported with n0 = 4, 8 or 16");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG((rhs_info.k0 == 2) || (rhs_info.k0 == 3), "Export to cl_image only supported with k0 = 4, 8 or 16");
        ARM_COMPUTE_RETURN_ERROR_ON_MSG(tensor_reshaped_info.data_type() != DataType::F32, "Export to cl_image only supported with F32 data type");
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
} // namespace cl_gemm
} // namespace arm_compute
