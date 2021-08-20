/*
 * Copyright (c) 2021 Arm Limited.
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
#ifndef SRC_COMMON_IOPERATOR_H_
#define SRC_COMMON_IOPERATOR_H_

#include "src/common/IContext.h"
#include "src/common/IQueue.h"

// TODO: Remove when all functions have been ported
#include "arm_compute/core/experimental/Types.h"
#include "arm_compute/runtime/IOperator.h"
#include "src/common/utils/Validate.h"

#include <vector>

struct AclOperator_
{
    arm_compute::detail::Header header{ arm_compute::detail::ObjectType::Operator, nullptr };

protected:
    AclOperator_()  = default;
    ~AclOperator_() = default;
};

namespace arm_compute
{
// Forward declarations
class ITensorPack;
namespace experimental
{
class IOperator;
} // namespace experimental

using MemoryRequirements = experimental::MemoryRequirements;

/** Base class specifying the operator interface */
class IOperator : public AclOperator_
{
public:
    /** Explict Operator Constructor
     *
     * @param[in] ctx Context to be used by the operator
     */
    explicit IOperator(IContext *ctx);
    /** Destructor */
    virtual ~IOperator();
    /** Checks if an operator is valid
     *
     * @return True if successful otherwise false
     */
    bool is_valid() const;
    /** Run the kernels contained in the function
     *
     * @param[in] queue   Queue to use
     * @param[in] tensors Vector that contains the tensors to operate on
     */
    virtual StatusCode run(IQueue &queue, ITensorPack &tensors);
    /** Run the kernels contained in the function
     *
     * @param[in] tensors Vector that contains the tensors to operate on
     */
    virtual StatusCode run(ITensorPack &tensors);
    /** Prepare the operator for execution
     *
     * Any one off pre-processing step required by the function is handled here
     *
     * @param[in] tensors Vector that contains the preparation tensors.
     *
     * @note Prepare stage might not need all the function's buffers' backing memory to be available in order to execute
     */
    virtual StatusCode prepare(ITensorPack &tensors);
    /** Return the memory requirements required by the workspace
     */
    virtual MemoryRequirements workspace() const;

    void set_internal_operator(std::unique_ptr<experimental::IOperator> op)
    {
        _op = std::move(op);
    }

private:
    std::unique_ptr<experimental::IOperator> _op{ nullptr };
};

/** Extract internal representation of an Operator
 *
 * @param[in] op Opaque operator pointer
 *
 * @return The internal representation as an IOperator
 */
inline IOperator *get_internal(AclOperator op)
{
    return static_cast<IOperator *>(op);
}

namespace detail
{
/** Check if an internal operator is valid
 *
 * @param[in] op Internal operator to check
 *
 * @return A status code
 */
inline StatusCode validate_internal_operator(const IOperator *op)
{
    if(op == nullptr || !op->is_valid())
    {
        ARM_COMPUTE_LOG_ERROR_ACL("[IOperator]: Invalid operator object");
        return StatusCode::InvalidArgument;
    }
    return StatusCode::Success;
}
} // namespace detail
} // namespace arm_compute
#endif /* SRC_COMMON_IOPERATOR_H_ */
