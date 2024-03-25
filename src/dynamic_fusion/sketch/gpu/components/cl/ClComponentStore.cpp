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
#include "ClComponentStore.h"

#include "src/dynamic_fusion/sketch/ArgumentPack.h"
#ifndef ACL_INTERNAL_TEST_CKW_IN_DF
#include "src/dynamic_fusion/sketch/gpu/template_writer/cl/ClTemplateStore.h"
#else //ACL_INTERNAL_TEST_CKW_IN_DF
#include "src/dynamic_fusion/sketch/gpu/ckw_driver/components/GpuCkwStore.h"
#endif //ACL_INTERNAL_TEST_CKW_IN_DF

#include <memory>

namespace arm_compute
{
namespace experimental
{
namespace dynamic_fusion
{
Status ClComponentStore::validate(const Properties &properties, const ArgumentPack<ITensorInfo> &tensors)
{
    ARM_COMPUTE_UNUSED(properties, tensors);
    return Status{};
}
ClComponentStore::ClComponentStore(ComponentId                      id,
                                   const Properties                &properties,
                                   const ArgumentPack<ITensorInfo> &tensors)
    : IGpuKernelComponent{id, properties, tensors},
#ifndef ACL_INTERNAL_TEST_CKW_IN_DF
      _component_writer{std::make_unique<ClTemplateStore>(id, tensors)}
#else  //ACL_INTERNAL_TEST_CKW_IN_DF
      _component_writer{std::make_unique<GpuCkwStore>(id, tensors)}
#endif //ACL_INTERNAL_TEST_CKW_IN_DF
{
}
ClComponentStore::~ClComponentStore()
{
}
#ifndef ACL_INTERNAL_TEST_CKW_IN_DF
const IGpuTemplateComponentWriter *ClComponentStore::template_writer() const
#else  //ACL_INTERNAL_TEST_CKW_IN_DF
const IGpuCkwComponentDriver *ClComponentStore::ckw_component_driver() const
#endif //ACL_INTERNAL_TEST_CKW_IN_DF
{
    return _component_writer.get();
}
} // namespace dynamic_fusion
} // namespace experimental
} // namespace arm_compute
