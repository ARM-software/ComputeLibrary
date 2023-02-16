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
#ifndef ARM_COMPUTE_CPU_KERNEL_SELECTION_TYPES_H
#define ARM_COMPUTE_CPU_KERNEL_SELECTION_TYPES_H

#include "arm_compute/core/Types.h"
#include "src/common/cpuinfo/CpuIsaInfo.h"

namespace arm_compute
{
namespace cpu
{
namespace kernels
{
// Selector data types
struct DataTypeISASelectorData
{
    DataType            dt;
    cpuinfo::CpuIsaInfo isa;
};

struct DataTypeDataLayoutISASelectorData
{
    DataType                   dt;
    DataLayout                 dl;
    const cpuinfo::CpuIsaInfo &isa;
};

struct CastDataTypeISASelectorData
{
    DataType                   src_dt;
    DataType                   dst_dt;
    const cpuinfo::CpuIsaInfo &isa;
};

struct PoolDataTypeISASelectorData
{
    DataType            dt;
    DataLayout          dl;
    int                 pool_stride_x;
    Size2D              pool_size;
    cpuinfo::CpuIsaInfo isa;
};

struct ElementwiseDataTypeISASelectorData
{
    DataType            dt;
    cpuinfo::CpuIsaInfo isa;
    int                 op;
};
struct DepthwiseConv2dNativeDataTypeISASelectorData
{
    DataType                   weights_dt;
    DataType                   source_dt;
    const cpuinfo::CpuIsaInfo &isa;
};

struct ActivationDataTypeISASelectorData
{
    DataType                                dt;
    const CPUModel                         &cpumodel;
    const cpuinfo::CpuIsaInfo              &isa;
    ActivationLayerInfo::ActivationFunction f;
};

struct CpuAddKernelDataTypeISASelectorData
{
    DataType            dt;
    cpuinfo::CpuIsaInfo isa;
    bool                can_use_fixedpoint;
};

struct ScaleKernelDataTypeISASelectorData
{
    DataType            dt;
    cpuinfo::CpuIsaInfo isa;
    InterpolationPolicy interpolation_policy;
};

// Selector pointer types
using DataTypeISASelectorPtr                      = std::add_pointer<bool(const DataTypeISASelectorData &data)>::type;
using DataTypeDataLayoutSelectorPtr               = std::add_pointer<bool(const DataTypeDataLayoutISASelectorData &data)>::type;
using PoolDataTypeISASelectorPtr                  = std::add_pointer<bool(const PoolDataTypeISASelectorData &data)>::type;
using ElementwiseDataTypeISASelectorPtr           = std::add_pointer<bool(const ElementwiseDataTypeISASelectorData &data)>::type;
using DepthwiseConv2dNativeDataTypeISASelectorPtr = std::add_pointer<bool(const DepthwiseConv2dNativeDataTypeISASelectorData &data)>::type;
using CastDataTypeISASelectorDataPtr              = std::add_pointer<bool(const CastDataTypeISASelectorData &data)>::type;
using ActivationDataTypeISASelectorDataPtr        = std::add_pointer<bool(const ActivationDataTypeISASelectorData &data)>::type;
using CpuAddKernelDataTypeISASelectorDataPtr      = std::add_pointer<bool(const CpuAddKernelDataTypeISASelectorData &data)>::type;
using ScaleKernelDataTypeISASelectorDataPtr       = std::add_pointer<bool(const ScaleKernelDataTypeISASelectorData &data)>::type;

} // namespace kernels
} // namespace cpu
} // namespace arm_compute

#endif // ARM_COMPUTE_CPU_KERNEL_SELECTION_TYPES_H
