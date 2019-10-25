/*
 * Copyright (c) 2019 ARM Limited.
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
#ifndef __ARM_COMPUTE_TEST_PARAMETERS_LIBRARY_H__
#define __ARM_COMPUTE_TEST_PARAMETERS_LIBRARY_H__

#include "arm_compute/runtime/IRuntimeContext.h"
#include "arm_compute/runtime/Tensor.h"
#if ARM_COMPUTE_CL
#include "arm_compute/runtime/CL/CLRuntimeContext.h"
#include "arm_compute/runtime/CL/CLTensor.h"
#endif /* ARM_COMPUTE_CL */
#ifdef ARM_COMPUTE_GC
#include "arm_compute/runtime/GLES_COMPUTE/GCRuntimeContext.h"
#include "arm_compute/runtime/GLES_COMPUTE/GCTensor.h"
#endif /* ARM_COMPUTE_GC */

#include <memory>

namespace arm_compute
{
namespace test
{
// Return type trait helper
template <class T>
struct ContextType
{
    using type = void;
};
template <>
struct ContextType<Tensor>
{
    using type = IRuntimeContext;
};

#if ARM_COMPUTE_CL
template <>
struct ContextType<CLTensor>
{
    using type = CLRuntimeContext;
};
#endif /* ARM_COMPUTE_CL */

#ifdef ARM_COMPUTE_GC
template <>
struct ContextType<GCTensor>
{
    using type = GCRuntimeContext;
};
#endif /* ARM_COMPUTE_GC */

/** Class that contains all the global parameters used by the tests */
class ParametersLibrary final
{
public:
    /** Default constructor */
    ParametersLibrary() = default;
    /** Set cpu context to be used by the tests
     *
     * @param[in] cpu_ctx CPU context to use
     */
    void set_cpu_ctx(std::unique_ptr<IRuntimeContext> cpu_ctx);
    /** Set gpu context to be used by the tests
     *
     * @param[in] cl_ctx GPU context to use
     */
    void set_cl_ctx(std::unique_ptr<IRuntimeContext> cl_ctx);
    /** Set gpu context to be used by the tests
     *
     * @param[in] gc_ctx GPU context to use
     */
    void set_gc_ctx(std::unique_ptr<IRuntimeContext> gc_ctx);
    /** Get context given a tensor type
     *
     * @tparam TensorType
     *
     * @return Pointer to the context
     */
    template <typename TensorType>
    typename ContextType<TensorType>::type *get_ctx()
    {
        return nullptr;
    }

private:
    std::unique_ptr<IRuntimeContext> _cpu_ctx{ nullptr };
    std::unique_ptr<IRuntimeContext> _cl_ctx{ nullptr };
    std::unique_ptr<IRuntimeContext> _gc_ctx{ nullptr };
};
} // namespace test
} // namespace arm_compute
#endif //__ARM_COMPUTE_TEST_PARAMETERS_LIBRARY_H__
