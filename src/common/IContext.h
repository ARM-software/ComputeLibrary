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
#ifndef SRC_COMMON_ICONTEXT_H
#define SRC_COMMON_ICONTEXT_H

#include "src/common/Types.h"
#include "src/common/utils/Log.h"
#include "src/common/utils/Object.h"

#include <atomic>

struct AclContext_
{
    arm_compute::detail::Header header{ arm_compute::detail::ObjectType::Context, nullptr };

protected:
    AclContext_()  = default;
    ~AclContext_() = default;
};

namespace arm_compute
{
// Forward declarations
class ITensorV2;
class IQueue;
class IOperator;

/**< Context interface */
class IContext : public AclContext_
{
public:
    IContext(Target target)
        : AclContext_(), _target(target), _refcount(0)
    {
    }
    /** Virtual Destructor */
    virtual ~IContext()
    {
        header.type = detail::ObjectType::Invalid;
    };
    /** Target type accessor
     *
     * @return Target that the context is associated with
     */
    Target type() const
    {
        return _target;
    }
    /** Increment context refcount */
    void inc_ref() const
    {
        ++_refcount;
    }
    /** Decrement context refcount */
    void dec_ref() const
    {
        --_refcount;
    }
    /** Reference counter accessor
     *
     * @return The number of references pointing to this object
     */
    int refcount() const
    {
        return _refcount;
    }
    /** Checks if an object is valid
     *
     * @return True if sucessful otherwise false
     */
    bool is_valid() const
    {
        return header.type == detail::ObjectType::Context;
    }
    /** Create a tensor object
     *
     * @param[in] desc     Descriptor to use
     * @param[in] allocate Flag to allocate tensor
     *
     * @return A pointer to the created tensor object
     */
    virtual ITensorV2 *create_tensor(const AclTensorDescriptor &desc, bool allocate) = 0;
    /** Create a queue object
     *
     * @param[in] options Queue options to be used
     *
     * @return A pointer to the created queue object
     */
    virtual IQueue *create_queue(const AclQueueOptions *options) = 0;
    virtual std::tuple<IOperator *, StatusCode> create_activation(const AclTensorDescriptor &src,
                                                                  const AclTensorDescriptor     &dst,
                                                                  const AclActivationDescriptor &act,
                                                                  bool                           is_validate) = 0;

private:
    Target                   _target;   /**< Target type of context */
    mutable std::atomic<int> _refcount; /**< Reference counter */
};

/** Extract internal representation of a Context
 *
 * @param[in] ctx Opaque context pointer
 *
 * @return The internal representation as an IContext
 */
inline IContext *get_internal(AclContext ctx)
{
    return static_cast<IContext *>(ctx);
}

namespace detail
{
/** Check if an internal context is valid
 *
 * @param[in] ctx Internal context to check
 *
 * @return A status code
 */
inline StatusCode validate_internal_context(const IContext *ctx)
{
    if(ctx == nullptr || !ctx->is_valid())
    {
        ARM_COMPUTE_LOG_ERROR_ACL("Invalid context object");
        return StatusCode::InvalidArgument;
    }
    return StatusCode::Success;
}
} // namespace detail
} // namespace arm_compute
#endif /* SRC_COMMON_ICONTEXT_H */
