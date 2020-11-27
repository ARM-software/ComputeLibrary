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
#ifndef ARM_COMPUTE_CLGEMMHELPERS_H
#define ARM_COMPUTE_CLGEMMHELPERS_H

#include "arm_compute/core/TensorInfo.h"
#include "arm_compute/core/Types.h"

namespace arm_compute
{
class ITensorInfo;
struct GEMMRHSMatrixInfo;

namespace cl_gemm
{
/** Configure @ref GEMMLHSMatrixInfo and @ref GEMMRHSMatrixInfo
 *
 * @param[in] m                  Number of rows (M) in the LHS matrix not reshaped
 * @param[in] n                  Number of columns (N) in the RHS matrix not reshaped
 * @param[in] m0                 Number of rows processed by each thread/work-item
 * @param[in] n0                 Number of columns processed by each thread/work-item
 * @param[in] k0                 Number of inner accumulation performed by each thread/work-item
 * @param[in] v0                 Number of vertical blocks of size (m0xk0) stored on the same output row
 * @param[in] h0                 Number of horizontal blocks of size (k0xn0) stored on the same output row
 * @param[in] lhs_interleave     True if the v0 (m0xk0) blocks have to be interleaved in the output row
 * @param[in] rhs_interleave     True if the h0 (k0xn0) blocks have to be interleaved in the output row
 * @param[in] lhs_transpose      True if the (m0xk0) block has to be transposed before been stored
 * @param[in] rhs_transpose      True if the (k0xn0) block has to be transposed before been stored
 * @param[in] export_to_cl_image (Optional) True if the RHS reshaped matrix has to be exported to cl_image
 *
 * @return @ref GEMMLHSMatrixInfo and @ref GEMMRHSMatrixInfo
 */
std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> configure_lhs_rhs_info(unsigned int m, unsigned int n, unsigned int m0, unsigned int n0, unsigned int k0, unsigned int v0, unsigned int h0,
                                                                       bool lhs_interleave, bool rhs_interleave, bool lhs_transpose, bool rhs_transpose, bool export_to_cl_image = false);

/** Select @ref GEMMLHSMatrixInfo and @ref GEMMRHSMatrixInfo
 *
 * This function accepts two pairs of GEMMLHSMatrixInfo/GEMMRHSMatrixInfo where only the first is with cl_image2d support,
 * and selects the valid one validating the GEMMRHSMatrixInfo. If the validation passes, the functions will return
 * the first GEMMLHSMatrixInfo/GEMMRHSMatrixInfo pair with cl_image2d support.
 *
 * @param[in] info_img  GEMMLHSMatrixInfo/GEMMRHSMatrixInfo with cl_image2d support
 * @param[in] info_buf  GEMMLHSMatrixInfo/GEMMRHSMatrixInfo to fall-back if cl_image2d cannot be used
 * @param[in] n         Number of columns (N) in the RHS matrix not reshaped
 * @param[in] k         Number of rows (K) in the RHS matrix not reshaped
 * @param[in] b         Batch size
 * @param[in] data_type Data type
 *
 * @return @ref GEMMLHSMatrixInfo and @ref GEMMRHSMatrixInfo
 */
std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> select_lhs_rhs_info(std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> info_img,
                                                                    std::pair<GEMMLHSMatrixInfo, GEMMRHSMatrixInfo> info_buf,
                                                                    unsigned int n, unsigned int k, unsigned int b, DataType data_type);

/** Update padding required to export the OpenCL buffer to OpenCL image2d
 *
 * @param[in,out] tensor ITensorInfo of the tensor required to be exported to OpenCL image2d
 */
void update_padding_for_cl_image(ITensorInfo *tensor);

/** Utility function to validate the image2d OpenCL object support on the RHS reshaped matrix
 *
 * @param[in] tensor_reshaped_info TensorInfo for the RHS reshaped matrix
 * @param[in] rhs_info             @ref GEMMRHSMatrixInfo
 *
 * @return Status reporting if we can use the image2d OpenCL object on the RHS reshaped matrix
 */
Status validate_image2d_support_on_rhs(const ITensorInfo &tensor_reshaped_info, const GEMMRHSMatrixInfo &rhs_info);
} // namespace cl_gemm
} // namespace arm_compute
#endif /*ARM_COMPUTE_CLGEMMHELPERS_H */
