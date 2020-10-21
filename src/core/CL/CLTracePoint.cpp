/*
 * Copyright (c) 2020 Arm Limited.
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
#include "arm_compute/core/TracePoint.h"

#include "arm_compute/core/CL/CLTypes.h"
#include "arm_compute/core/CL/ICLArray.h"
#include "arm_compute/core/CL/ICLDistribution1D.h"
#include "arm_compute/core/CL/ICLHOG.h"
#include "arm_compute/core/CL/ICLLut.h"
#include "arm_compute/core/CL/ICLMultiHOG.h"
#include "arm_compute/core/CL/ICLMultiImage.h"
#include "arm_compute/core/CL/ICLTensor.h"
#include "utils/TypePrinter.h"

#include <vector>

namespace arm_compute
{
std::string to_string(const ICLTensor &arg)
{
    std::stringstream str;
    str << "TensorInfo(" << *arg.info() << ")";
    return str.str();
}

template <>
TracePoint::Args &&operator<<(TracePoint::Args &&tp, const ICLTensor *arg)
{
    tp.args.push_back("ICLTensor(" + to_string_if_not_null(arg) + ")");
    return std::move(tp);
}

TRACE_TO_STRING(std::vector<ICLTensor *>)
TRACE_TO_STRING(ICLMultiImage)
TRACE_TO_STRING(ICLDetectionWindowArray)
TRACE_TO_STRING(ICLKeyPointArray)
TRACE_TO_STRING(ICLLKInternalKeypointArray)
TRACE_TO_STRING(ICLCoefficientTableArray)
TRACE_TO_STRING(ICLCoordinates2DArray)
TRACE_TO_STRING(ICLOldValArray)
TRACE_TO_STRING(cl::Buffer)
TRACE_TO_STRING(ICLDistribution1D)
TRACE_TO_STRING(ICLMultiHOG)
TRACE_TO_STRING(ICLHOG)
TRACE_TO_STRING(ICLLut)
TRACE_TO_STRING(ICLSize2DArray)
TRACE_TO_STRING(std::vector<const ICLTensor *>)

CONST_PTR_CLASS(std::vector<ICLTensor *>)
CONST_PTR_CLASS(ICLMultiImage)
CONST_PTR_CLASS(ICLDetectionWindowArray)
CONST_PTR_CLASS(ICLKeyPointArray)
CONST_PTR_CLASS(ICLLKInternalKeypointArray)
CONST_PTR_CLASS(ICLCoefficientTableArray)
CONST_PTR_CLASS(ICLCoordinates2DArray)
CONST_PTR_CLASS(ICLOldValArray)
CONST_PTR_CLASS(cl::Buffer)
CONST_PTR_CLASS(ICLDistribution1D)
CONST_PTR_CLASS(ICLMultiHOG)
CONST_PTR_CLASS(ICLHOG)
CONST_PTR_CLASS(ICLLut)
CONST_PTR_CLASS(ICLSize2DArray)
CONST_PTR_CLASS(std::vector<const ICLTensor *>)
} // namespace arm_compute
