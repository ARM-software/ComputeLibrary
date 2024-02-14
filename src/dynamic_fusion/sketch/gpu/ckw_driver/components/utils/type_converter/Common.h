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
#ifndef ACL_SRC_DYNAMIC_FUSION_SKETCH_GPU_CKW_DRIVER_COMPONENTS_UTILS_TYPE_CONVERTER_COMMON_H
#define ACL_SRC_DYNAMIC_FUSION_SKETCH_GPU_CKW_DRIVER_COMPONENTS_UTILS_TYPE_CONVERTER_COMMON_H

#include "arm_compute/core/CoreTypes.h"
#include "arm_compute/core/ITensorInfo.h"
#include "arm_compute/core/TensorShape.h"

#include "src/dynamic_fusion/sketch/gpu/GpuKernelArgument.h"

#include "compute_kernel_writer/include/ckw/TensorInfo.h"
#include "compute_kernel_writer/include/ckw/types/DataType.h"
#include "compute_kernel_writer/include/ckw/types/TensorComponentType.h"
#include "compute_kernel_writer/include/ckw/types/TensorStorageType.h"

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
/** Convert the Compute Library data type to Compute Kernel Writer data type
 *
 * @param[in] dt The Compute Library data type
 *
 * @return the Compute Kernel Writer data type (ckw::DataType)
 */
ckw::DataType to_ckw(DataType dt);

/** Convert the Compute Library tensor shape to Compute Kernel Writer tensor shape
 *
 * @param[in] shape The Compute Library tensor shape
 *
 * @return the Compute Kernel Writer tensor shape (ckw::TensorShape)
 */
ckw::TensorShape to_ckw(const TensorShape &shape);

/** Convert the Compute Library data layout to Compute Kernel Writer data layout
 *
 * @param[in] dl The Compute Library data layout
 *
 * @return the Compute Kernel Writer data layout (ckw::TensorDataLayout)
 */
ckw::TensorDataLayout to_ckw(DataLayout dl);

/** Convert the Compute Library tensor info to Compute Kernel Writer tensor info
 *
 * @param[in] tensor_info The Compute Library tensor info
 *
 * @return the Compute Kernel Writer tensor info (ckw::TensorInfo)
 */
ckw::TensorInfo to_ckw(const ITensorInfo &tensor_info);

/** Convert the Compute Library tensor storage to Compute Kernel Writer tensor storage
 *
 * @param[in] storage The Compute Library tensor storage
 *
 * @return the Compute Kernel Writer tensor storate (ckw::TensorStorageType)
 */
ckw::TensorStorageType to_ckw(const TensorStorageType &storage);

/** Convert the Compute Kernel Writer tensor component to Compute Library tensor component
 *
 * @param[in] component The Compute Kernel Writer tensor component
 *
 * @return the Compute Library tensor component
 */
TensorComponentType from_ckw(const ckw::TensorComponentType &component);

/** Convert the Compute Kernel Writer tensor storage to Compute Library tensor storage
 *
 * @param[in] storage The Compute Kernel Writer tensor storage
 *
 * @return the Compute Library tensor storage
 */
TensorStorageType from_ckw(const ckw::TensorStorageType &storage);

} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
#endif // ACL_SRC_DYNAMIC_FUSION_SKETCH_GPU_CKW_DRIVER_COMPONENTS_UTILS_TYPE_CONVERTER_COMMON_H
