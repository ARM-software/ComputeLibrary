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

/** Structure to capture internally memory requirements */
struct MemoryInfo
{
    MemoryInfo(AclTensorSlot slot_id, size_t size, size_t alignment) noexcept
        : slot_id(slot_id),
          size(size),
          alignment(alignment)
    {
    }
    AclTensorSlot slot_id;
    size_t        size;
    size_t        alignment;
};
using MemoryRequirements = std::vector<MemoryInfo>;

/** Base class specifying the operator interface */
class IOperator : public AclOperator_
{
public:
    /** Explict Operator Constructor
     *
     * @param[in] ctx Context to be used by the operator
     */
    explicit IOperator(IContext *ctx)
    {
        this->header.ctx = ctx;
        this->header.ctx->inc_ref();
    }

    /** Destructor */
    virtual ~IOperator()
    {
        this->header.ctx->dec_ref();
        this->header.type = detail::ObjectType::Invalid;
    };

    /** Checks if an operator is valid
     *
     * @return True if successful otherwise false
     */
    bool is_valid() const
    {
        return this->header.type == detail::ObjectType::Operator;
    };

    /** Run the kernels contained in the function
     *
     * @param[in] queue   Queue to run a kernel on
     * @param[in] tensors Vector that contains the tensors to operate on
     */
    virtual StatusCode run(IQueue &queue, ITensorPack &tensors) = 0;

    /** Prepare the operator for execution
     *
     * Any one off pre-processing step required by the function is handled here
     *
     * @param[in] constants Vector that contains the constants tensors.
     *
     * @note Prepare stage might not need all the function's buffers' backing memory to be available in order to execute
     */
    virtual StatusCode prepare(ITensorPack &constants)
    {
        ARM_COMPUTE_UNUSED(constants);
        return StatusCode::Success;
    }

    /** Return the memory requirements required by the workspace
     */
    virtual MemoryRequirements workspace() const
    {
        return {};
    }
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
