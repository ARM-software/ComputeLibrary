/*
 * Copyright (c) 2023-2024 Arm Limited.
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
#ifndef ACL_SRC_DYNAMIC_FUSION_SKETCH_GPU_CKW_DRIVER_COMPONENTS_UTILS_CKWHELPER_H
#define ACL_SRC_DYNAMIC_FUSION_SKETCH_GPU_CKW_DRIVER_COMPONENTS_UTILS_CKWHELPER_H

#include "src/dynamic_fusion/sketch/gpu/ckw_driver/GpuCkwScopedKernelWriter.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
/** Get coordinate along one axis.
 *
 * @param[in,out] writer Writer
 * @param[out]    coord  Resultant coordinate
 * @param[in]     gid    Global work item id
 * @param[in]     step   Step size / vector size
 */
void get_coordinate_from_gws(GpuCkwScopedKernelWriter writer,
                             ckw::TileOperand        &coord,
                             const ckw::TileOperand  &gid,
                             ckw::TileOperand        &step);

/** Get boundary aware coordinate along one axis.
 *
 * @param[in,out] writer     Writer
 * @param[out]    coord      Resultant coordinate
 * @param[in]     gid        Global work item id
 * @param[in]     step       Step size / vector size
 * @param[in]     shift_back It is (step - leftover_step) % step
 * @param[in]     const_0    Constant tile of value 0
 */
void get_coordinate_from_gws_overlapping_min(GpuCkwScopedKernelWriter writer,
                                             ckw::TileOperand        &coord,
                                             const ckw::TileOperand  &gid,
                                             ckw::TileOperand        &step,
                                             ckw::TileOperand        &shift_back,
                                             ckw::TileOperand        &const_0);
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif // ACL_SRC_DYNAMIC_FUSION_SKETCH_GPU_CKW_DRIVER_COMPONENTS_UTILS_CKWHELPER_H
