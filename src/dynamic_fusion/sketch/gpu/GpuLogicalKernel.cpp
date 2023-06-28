/*
 * Copyright (c) 2022-2023 Arm Limited.
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
#include "GpuLogicalKernel.h"

#include "arm_compute/core/experimental/Types.h"

#include "src/dynamic_fusion/sketch/ArgumentPack.h"
#include "src/dynamic_fusion/sketch/gpu/GpuComponentServices.h"
#include "src/dynamic_fusion/sketch/gpu/components/IGpuKernelComponent.h"
#include "src/dynamic_fusion/sketch/gpu/components/cl/ClComponentStore.h"
#ifndef ACL_INTERNAL_TEST_CKW_IN_DF
#include "src/dynamic_fusion/sketch/gpu/template_writer/cl/ClTemplateWriter.h"
#else // ACL_INTERNAL_TEST_CKW_IN_DF
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/GpuCkwDriver.h"
#endif // ACL_INTERNAL_TEST_CKW_IN_DF

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
GpuLogicalKernel::GpuLogicalKernel(GpuComponentServices *services, const GpuKernelComponentGroup &components)
    : _comp_group{ components }, _store_components{}
{
    ARM_COMPUTE_UNUSED(services);
}

GpuKernelSourceCode GpuLogicalKernel::write_kernel_code()
{
    GpuKernelSourceCode code;
#ifndef ACL_INTERNAL_TEST_CKW_IN_DF
    ClTemplateWriter writer { _comp_group };
#else  // ACL_INTERNAL_TEST_CKW_IN_DF
    GpuCkwDriver writer { _comp_group };
#endif // ACL_INTERNAL_TEST_CKW_IN_DF

    code.name(writer.get_name());
    code.code(writer.get_code());
#ifndef ACL_INTERNAL_TEST_CKW_IN_DF
    code.arguments(writer.get_tensors());
#else  // ACL_INTERNAL_TEST_CKW_IN_DF
    code.arguments(writer.get_kernel_arguments());
#endif // ACL_INTERNAL_TEST_CKW_IN_DF
    code.build_options(writer.get_build_options());
    code.config_id(writer.get_config_id());
    code.window(writer.get_window());

    return code;
}
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
