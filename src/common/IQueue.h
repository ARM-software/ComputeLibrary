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
#ifndef SRC_COMMON_IQUEUE_H_
#define SRC_COMMON_IQUEUE_H_

#include "src/common/IContext.h"

struct AclQueue_
{
    arm_compute::detail::Header header{ arm_compute::detail::ObjectType::Queue, nullptr };

protected:
    AclQueue_()  = default;
    ~AclQueue_() = default;
};

namespace arm_compute
{
/** Base class specifying the queue interface */
class IQueue : public AclQueue_
{
public:
    /** Explict Operator Constructor
     *
     * @param[in] ctx Context to be used by the operator
     */
    explicit IQueue(IContext *ctx)
    {
        this->header.ctx = ctx;
        this->header.ctx->inc_ref();
    }
    /** Destructor */
    virtual ~IQueue()
    {
        this->header.ctx->dec_ref();
        this->header.type = detail::ObjectType::Invalid;
    };
    /** Checks if a queue is valid
     *
     * @return True if successful otherwise false
     */
    bool is_valid() const
    {
        return this->header.type == detail::ObjectType::Queue;
    };
    virtual StatusCode finish() = 0;
};

/** Extract internal representation of a Queue
 *
 * @param[in] queue Opaque queue pointer
 *
 * @return The internal representation as an IQueue
 */
inline IQueue *get_internal(AclQueue queue)
{
    return static_cast<IQueue *>(queue);
}

namespace detail
{
/** Check if an internal queue is valid
 *
 * @param[in] queue Internal queue to check
 *
 * @return A status code
 */
inline StatusCode validate_internal_queue(const IQueue *queue)
{
    if(queue == nullptr || !queue->is_valid())
    {
        ARM_COMPUTE_LOG_ERROR_ACL("[IQueue]: Invalid queue object");
        return StatusCode::InvalidArgument;
    }
    return StatusCode::Success;
}
} // namespace detail
} // namespace arm_compute
#endif /* SRC_COMMON_IQUEUE_H_ */
