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
#ifndef ARM_COMPUTE_ACL_HPP_
#define ARM_COMPUTE_ACL_HPP_

#include "arm_compute/Acl.h"

#include <cstdlib>
#include <memory>
#include <string>

#if defined(ARM_COMPUTE_EXCEPTIONS_ENABLED)
#include <exception>
#endif /* defined(ARM_COMPUTE_EXCEPTIONS_ENABLED) */

// Helper Macros
#define ARM_COMPUTE_IGNORE_UNUSED(x) (void)(x)

namespace acl
{
// Forward declarations
class Context;

/**< Status code enum */
enum class StatusCode
{
    Success            = AclSuccess,
    RuntimeError       = AclRuntimeError,
    OutOfMemory        = AclOutOfMemory,
    Unimplemented      = AclUnimplemented,
    UnsupportedTarget  = AclUnsupportedTarget,
    InvalidArgument    = AclInvalidArgument,
    InvalidTarget      = AclInvalidTarget,
    UnsupportedConfig  = AclUnsupportedConfig,
    InvalidObjectState = AclInvalidObjectState,
};

/**< Utility namespace containing helpers functions */
namespace detail
{
/** Construct to handle destruction of objects
 *
 * @tparam T Object base type
 */
template <typename T>
struct ObjectDeleter
{
};

#define OBJECT_DELETER(obj, func)              \
    template <>                                \
    struct ObjectDeleter<obj>                  \
        \
    {                                          \
        static inline AclStatus Destroy(obj v) \
        {                                      \
            return func(v);                    \
        }                                      \
    };

OBJECT_DELETER(AclContext, AclDestroyContext)

#undef OBJECT_DELETER

/** Convert a strongly typed enum to an old plain c enum
  *
  * @tparam E  Plain old C enum
  * @tparam SE Strongly typed resulting enum
  *
  * @param[in] v Value to convert
  *
  * @return A corresponding plain old C enumeration
  */
template <typename E, typename SE>
constexpr E as_cenum(SE v) noexcept
{
    return static_cast<E>(static_cast<typename std::underlying_type<SE>::type>(v));
}

/** Convert plain old enumeration to a strongly typed enum
  *
  * @tparam SE Strongly typed resulting enum
  * @tparam E  Plain old C enum
  *
  * @param[in] val Value to convert
  *
  * @return A corresponding strongly typed enumeration
  */
template <typename SE, typename E>
constexpr SE as_enum(E val) noexcept
{
    return static_cast<SE>(val);
}

/** Object base class for library objects
 *
 * Class is defining basic common interface for all the library objects
 *
 * @tparam T Object type to be templated on
 */
template <typename T>
class ObjectBase
{
public:
    /** Destructor */
    ~ObjectBase() = default;
    /** Copy constructor */
    ObjectBase(const ObjectBase<T> &) = default;
    /** Move Constructor */
    ObjectBase(ObjectBase<T> &&) = default;
    /** Copy assignment operator */
    ObjectBase<T> &operator=(const ObjectBase<T> &) = default;
    /** Move assignment operator */
    ObjectBase<T> &operator=(ObjectBase<T> &&) = default;
    /** Reset object value
     *
     * @param [in] val Value to set
     */
    void reset(T *val)
    {
        _object.reset(val, detail::ObjectDeleter<T *>::Destroy);
    }
    /** Access uderlying object
     *
     * @return Underlying object
     */
    const T *get() const
    {
        return _object.get();
    }
    /** Access uderlying object
     *
     * @return Underlying object
     */
    T *get()
    {
        return _object.get();
    }

protected:
    /** Constructor */
    ObjectBase() = default;

protected:
    std::shared_ptr<T> _object{ nullptr }; /**< Library object */
};

/** Equality operator for library object
 *
 * @tparam T Parameter to template on
 *
 * @param[in] lhs Left hand-side argument
 * @param[in] rhs Right hand-side argument
 *
 * @return True if objects are equal, else false
 */
template <typename T>
bool operator==(const ObjectBase<T> &lhs, const ObjectBase<T> &rhs)
{
    return lhs.get() == rhs.get();
}

/** Inequality operator for library object
 *
 * @tparam T Parameter to template on
 *
 * @param[in] lhs Left hand-side argument
 * @param[in] rhs Right hand-side argument
 *
 * @return True if objects are equal, else false
 */
template <typename T>
bool operator!=(const ObjectBase<T> &lhs, const ObjectBase<T> &rhs)
{
    return !(lhs == rhs);
}
} // namespace detail

#if defined(ARM_COMPUTE_EXCEPTIONS_ENABLED)
/** Status class
 *
 * Class is an extension of std::exception and contains the underlying
 * status construct and an error explanatory message to be reported.
 *
 * @note Class is visible only when exceptions are enabled during compilation
 */
class Status : public std::exception
{
public:
    /** Constructor
     *
     * @param[in] status Status returned
     * @param[in] msg    Error message to be bound with the exception
     */
    Status(StatusCode status, const std::string &msg)
        : _status(status), _msg(msg)
    {
    }
    /** Returns an explanatory exception message
     *
     * @return Status message
     */
    const char *what() const noexcept override
    {
        return _msg.c_str();
    }
    /** Underlying status accessor
     *
     * @return Status code
     */
    StatusCode status() const
    {
        return _status;
    }
    /** Explicit status converter
     *
     * @return Status code
     */
    explicit operator StatusCode() const
    {
        return _status;
    }

private:
    StatusCode  _status; /**< Status code */
    std::string _msg;    /**< Status message */
};

/** Reports an error status and throws an exception object in case of failure
 *
 * @note This implementation is used when exceptions are enabled during compilation
 *
 * @param[in] status Status to report
 * @param[in] msg    Explanatory error messaged
 *
 * @return Status code
 */
static inline StatusCode report_status(StatusCode status, const std::string &msg)
{
    if(status != StatusCode::Success)
    {
        throw Status(status, msg);
    }
    return status;
}
#else  /* defined(ARM_COMPUTE_EXCEPTIONS_ENABLED) */
/** Reports a status code
 *
 * @note This implementation is used when exceptions are disabled during compilation
 * @note Message is surpressed and not reported in this case
 *
 * @param[in] status Status to report
 * @param[in] msg    Explanatory error messaged
 *
 * @return Status code
 */
static inline StatusCode report_status(StatusCode status, const std::string &msg)
{
    ARM_COMPUTE_IGNORE_UNUSED(msg);
    return status;
}
#endif /* defined(ARM_COMPUTE_EXCEPTIONS_ENABLED) */

/**< Target enum */
enum class Target
{
    Cpu    = AclCpu,   /**< Cpu target that leverages SIMD */
    GpuOcl = AclGpuOcl /**< Gpu target that leverages OpenCL */
};

/**< Available execution modes */
enum class ExecutionMode
{
    FastRerun = AclPreferFastRerun, /**< Prefer minimum latency in consecutive runs, might introduce higher startup times */
    FastStart = AclPreferFastStart, /**< Prefer minimizing startup time */
};

/** Context class
 *
 * Context acts as a central aggregate service for further objects created from it.
 * It provides, internally, common facilities in order to avoid the use of global
 * statically initialized objects that can lead to important side-effect under
 * specific execution contexts.
 *
 * For example context contains allocators for object creation, for further backing memory allocation,
 * any serialization interfaces and other modules that affect the construction of objects,
 * like program caches for OpenCL.
 */
class Context : public detail::ObjectBase<AclContext_>
{
public:
    /**< Context options */
    struct Options
    {
        /** Default Constructor
         *
         * @note By default no precision loss is enabled for operators
         * @note By default the preferred execution mode is to favor multiple consecutive reruns of an operator
         */
        Options() = default;
        /** Constructor
         *
         * @param[in] mode              Execution mode to be used
         * @param[in] caps              Capabilities to be used
         * @param[in] enable_fast_math  Allow precision loss in favor of performance
         * @param[in] kernel_config     Kernel configuration file containing construction tuning meta-data
         * @param[in] max_compute_units Max compute units that are expected to used
         * @param[in] allocator         Allocator to be used for internal memory allocation
         */
        Options(ExecutionMode         mode,
                AclTargetCapabilities caps,
                bool                  enable_fast_math,
                const char           *kernel_config,
                int32_t               max_compute_units,
                AclAllocator         *allocator)
        {
            opts.mode               = detail::as_cenum<AclExecutionMode>(mode);
            opts.capabilities       = caps;
            opts.enable_fast_math   = enable_fast_math;
            opts.kernel_config_file = kernel_config;
            opts.max_compute_units  = max_compute_units;
            opts.allocator          = allocator;
        }
        AclContextOptions opts{ acl_default_ctx_options };
    };

public:
    /** Constructor
     *
     * @note Serves as a simpler delegate constructor
     * @note As context options, default conservative options will be used
     *
     * @param[in]  target Target to create context for
     * @param[out] status Status information if requested
     */
    explicit Context(Target target, StatusCode *status = nullptr)
        : Context(target, Options(), status)
    {
    }
    /** Constructor
     *
     * @param[in]  target  Target to create context for
     * @param[in]  options Context construction options
     * @param[out] status  Status information if requested
     */
    Context(Target target, const Options &options, StatusCode *status = nullptr)
    {
        AclContext ctx;
        const auto st = detail::as_enum<StatusCode>(AclCreateContext(&ctx, detail::as_cenum<AclTarget>(target), &options.opts));
        reset(ctx);
        report_status(st, "Failure during context creation");
        if(status)
        {
            *status = st;
        }
    }
};
} // namespace acl
#undef ARM_COMPUTE_IGNORE_UNUSED
#endif /* ARM_COMPUTE_ACL_HPP_ */
