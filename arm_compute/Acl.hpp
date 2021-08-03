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
#include <vector>

#if defined(ARM_COMPUTE_EXCEPTIONS_ENABLED)
#include <exception>
#endif /* defined(ARM_COMPUTE_EXCEPTIONS_ENABLED) */

// Helper Macros
#define ARM_COMPUTE_IGNORE_UNUSED(x) (void)(x)

namespace acl
{
// Forward declarations
class Context;
class Queue;
class Tensor;
class TensorPack;

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
OBJECT_DELETER(AclQueue, AclDestroyQueue)
OBJECT_DELETER(AclTensor, AclDestroyTensor)
OBJECT_DELETER(AclTensorPack, AclDestroyTensorPack)
OBJECT_DELETER(AclOperator, AclDestroyOperator)

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
static inline void report_status(StatusCode status, const std::string &msg)
{
    if(status != StatusCode::Success)
    {
        throw Status(status, msg);
    }
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
static inline void report_status(StatusCode status, const std::string &msg)
{
    ARM_COMPUTE_IGNORE_UNUSED(status);
    ARM_COMPUTE_IGNORE_UNUSED(msg);
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
        static constexpr int32_t num_threads_auto = -1; /**< Allow runtime to specify number of threads */

        /** Default Constructor
         *
         * @note By default no precision loss is enabled for operators
         * @note By default the preferred execution mode is to favor multiple consecutive reruns of an operator
         */
        Options()
            : Options(ExecutionMode::FastRerun /* mode */,
                      AclCpuCapabilitiesAuto /* caps */,
                      false /* enable_fast_math */,
                      nullptr /* kernel_config */,
                      num_threads_auto /* max_compute_units */,
                      nullptr /* allocator */)
        {
        }
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
            copts.mode               = detail::as_cenum<AclExecutionMode>(mode);
            copts.capabilities       = caps;
            copts.enable_fast_math   = enable_fast_math;
            copts.kernel_config_file = kernel_config;
            copts.max_compute_units  = max_compute_units;
            copts.allocator          = allocator;
        }

        AclContextOptions copts{};
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
        const auto st = detail::as_enum<StatusCode>(AclCreateContext(&ctx, detail::as_cenum<AclTarget>(target), &options.copts));
        reset(ctx);
        report_status(st, "[Compute Library] Failed to create context");
        if(status)
        {
            *status = st;
        }
    }
};

/**< Available tuning modes */
enum class TuningMode
{
    Rapid      = AclRapid,
    Normal     = AclNormal,
    Exhaustive = AclExhaustive
};

/** Queue class
 *
 * Queue is responsible for the execution related aspects, with main responsibilities those of
 * scheduling and tuning operators.
 *
 * Multiple queues can be created from the same context, and the same operator can be scheduled on each concurrently.
 *
 * @note An operator might depend on the maximum possible compute units that are provided in the context,
 *       thus in cases where the number of the scheduling units of the queue are greater might lead to errors.
 */
class Queue : public detail::ObjectBase<AclQueue_>
{
public:
    /**< Queue options */
    struct Options
    {
        /** Default Constructor
         *
         * As default options, no tuning will be performed, and the number of scheduling units will
         * depends on internal device discovery functionality
         */
        Options()
            : opts{ AclTuningModeNone, 0 } {};
        /** Constructor
         *
         * @param[in] mode          Tuning mode to be used
         * @param[in] compute_units Number of scheduling units to be used
         */
        Options(TuningMode mode, int32_t compute_units)
            : opts{ detail::as_cenum<AclTuningMode>(mode), compute_units }
        {
        }

        AclQueueOptions opts;
    };

public:
    /** Constructor
     *
     * @note Serves as a simpler delegate constructor
     * @note As queue options, default conservative options will be used
     *
     * @param[in]  ctx    Context to create queue for
     * @param[out] status Status information if requested
     */
    explicit Queue(Context &ctx, StatusCode *status = nullptr)
        : Queue(ctx, Options(), status)
    {
    }
    /** Constructor
     *
     * @note As queue options, default conservative options will be used
     *
     * @param[in]  ctx     Context from where the queue will be created from
     * @param[in]  options Queue options to be used
     * @param[out] status  Status information if requested
     */
    explicit Queue(Context &ctx, const Options &options = Options(), StatusCode *status = nullptr)
    {
        AclQueue   queue;
        const auto st = detail::as_enum<StatusCode>(AclCreateQueue(&queue, ctx.get(), &options.opts));
        reset(queue);
        report_status(st, "[Compute Library] Failed to create queue!");
        if(status)
        {
            *status = st;
        }
    }
    /** Block until all the tasks of the queue have been marked as finished
     *
     * @return Status code
     */
    StatusCode finish()
    {
        return detail::as_enum<StatusCode>(AclQueueFinish(_object.get()));
    }
};

/**< Data type enumeration */
enum class DataType
{
    Unknown  = AclDataTypeUnknown,
    UInt8    = AclUInt8,
    Int8     = AclInt8,
    UInt16   = AclUInt16,
    Int16    = AclInt16,
    UInt32   = AclUint32,
    Int32    = AclInt32,
    Float16  = AclFloat16,
    BFloat16 = AclBFloat16,
    Float32  = AclFloat32,
};

/** Tensor Descriptor class
 *
 * Structure that contains all the required meta-data to represent a tensor
 */
class TensorDescriptor
{
public:
    /** Constructor
     *
     * @param[in] shape Shape of the tensor
     * @param[in] data_type Data type of the tensor
     */
    TensorDescriptor(const std::vector<int32_t> &shape, DataType data_type)
        : _shape(shape), _data_type(data_type)
    {
        _cdesc.ndims     = _shape.size();
        _cdesc.shape     = _shape.data();
        _cdesc.data_type = detail::as_cenum<AclDataType>(_data_type);
        _cdesc.strides   = nullptr;
        _cdesc.boffset   = 0;
    }
    /** Constructor
     *
     * @param[in] desc C-type descriptor
     */
    explicit TensorDescriptor(const AclTensorDescriptor &desc)
    {
        _cdesc     = desc;
        _data_type = detail::as_enum<DataType>(desc.data_type);
        _shape.reserve(desc.ndims);
        for(int32_t d = 0; d < desc.ndims; ++d)
        {
            _shape.emplace_back(desc.shape[d]);
        }
    }
    /** Get underlying C tensor descriptor
     *
     * @return Underlying structure
     */
    const AclTensorDescriptor *get() const
    {
        return &_cdesc;
    }
    /** Operator to compare two TensorDescriptor
     *
     * @param[in] other The instance to compare against
     *
     * @return True if two instances have the same shape and data type
     */
    bool operator==(const TensorDescriptor &other)
    {
        bool is_same = true;

        is_same &= _data_type == other._data_type;
        is_same &= _shape.size() == other._shape.size();

        if(is_same)
        {
            for(uint32_t d = 0; d < _shape.size(); ++d)
            {
                is_same &= _shape[d] == other._shape[d];
            }
        }

        return is_same;
    }

private:
    std::vector<int32_t> _shape{};
    DataType             _data_type{};
    AclTensorDescriptor  _cdesc{};
};

/** Import memory types */
enum class ImportType
{
    Host = AclImportMemoryType::AclHostPtr
};

/** Tensor class
 *
 * Tensor is an mathematical construct that can represent an N-Dimensional space.
 *
 * @note Maximum dimensionality support is 6 internally at the moment
 */
class Tensor : public detail::ObjectBase<AclTensor_>
{
public:
    /** Constructor
     *
     * @note Tensor memory is allocated
     *
     * @param[in]  ctx    Context from where the tensor will be created from
     * @param[in]  desc   Tensor descriptor to be used
     * @param[out] status Status information if requested
     */
    Tensor(Context &ctx, const TensorDescriptor &desc, StatusCode *status = nullptr)
        : Tensor(ctx, desc, true, status)
    {
    }
    /** Constructor
     *
     * @param[in]  ctx    Context from where the tensor will be created from
     * @param[in]  desc   Tensor descriptor to be used
     * @param[in]  allocate Flag to indicate if the tensor needs to be allocated
     * @param[out] status Status information if requested
     */
    Tensor(Context &ctx, const TensorDescriptor &desc, bool allocate, StatusCode *status)
    {
        AclTensor  tensor;
        const auto st = detail::as_enum<StatusCode>(AclCreateTensor(&tensor, ctx.get(), desc.get(), allocate));
        reset(tensor);
        report_status(st, "[Compute Library] Failed to create tensor!");
        if(status)
        {
            *status = st;
        }
    }
    /** Maps the backing memory of a given tensor that can be used by the host to access any contents
     *
     * @return A valid non-zero pointer in case of success else nullptr
     */
    void *map()
    {
        void      *handle = nullptr;
        const auto st     = detail::as_enum<StatusCode>(AclMapTensor(_object.get(), &handle));
        report_status(st, "[Compute Library] Failed to map the tensor and extract the tensor's backing memory!");
        return handle;
    }
    /** Unmaps tensor's memory
     *
     * @param[in] handle Handle to unmap
     *
     * @return Status code
     */
    StatusCode unmap(void *handle)
    {
        const auto st = detail::as_enum<StatusCode>(AclUnmapTensor(_object.get(), handle));
        report_status(st, "[Compute Library] Failed to unmap the tensor!");
        return st;
    }
    /** Import external memory to a given tensor object
     *
     * @param[in] handle External memory handle
     * @param[in] type   Type of memory to be imported
     *
     * @return Status code
     */
    StatusCode import(void *handle, ImportType type)
    {
        const auto st = detail::as_enum<StatusCode>(AclTensorImport(_object.get(), handle, detail::as_cenum<AclImportMemoryType>(type)));
        report_status(st, "[Compute Library] Failed to import external memory to tensor!");
        return st;
    }
    /** Get the size of the tensor in byte
     *
     * @note The size isn't based on allocated memory, but based on information in its descriptor (dimensions, data type, etc.).
     *
     * @return The size of the tensor in byte
     */
    uint64_t get_size()
    {
        uint64_t   size{ 0 };
        const auto st = detail::as_enum<StatusCode>(AclGetTensorSize(_object.get(), &size));
        report_status(st, "[Compute Library] Failed to get the size of the tensor");
        return size;
    }
    /** Get the descriptor of this tensor
     *
     * @return The descriptor describing the characteristics of this tensor
     */
    TensorDescriptor get_descriptor()
    {
        AclTensorDescriptor desc;
        const auto          st = detail::as_enum<StatusCode>(AclGetTensorDescriptor(_object.get(), &desc));
        report_status(st, "[Compute Library] Failed to get the descriptor of the tensor");
        return TensorDescriptor(desc);
    }
};

/** Tensor pack class
 *
 * Pack is a utility construct that is used to create a collection of tensors that can then
 * be passed into operator as inputs.
 */
class TensorPack : public detail::ObjectBase<AclTensorPack_>
{
public:
    /** Pack pair construct */
    struct PackPair
    {
        /** Constructor
         *
         * @param[in] tensor_ Tensor to pack
         * @param[in] slot_id_ Slot identification of the tensor in respect with the operator
         */
        PackPair(Tensor *tensor_, int32_t slot_id_)
            : tensor(tensor_), slot_id(slot_id_)
        {
        }

        Tensor *tensor{ nullptr };         /**< Tensor object */
        int32_t slot_id{ AclSlotUnknown }; /**< Slot id in respect with the operator */
    };

public:
    /** Constructor
     *
     * @param[in]  ctx    Context from where the tensor pack will be created from
     * @param[out] status Status information if requested
     */
    explicit TensorPack(Context &ctx, StatusCode *status = nullptr)
    {
        AclTensorPack pack;
        const auto    st = detail::as_enum<StatusCode>(AclCreateTensorPack(&pack, ctx.get()));
        reset(pack);
        report_status(st, "[Compute Library] Failure during tensor pack creation");
        if(status)
        {
            *status = st;
        }
    }
    /** Add tensor to tensor pack
     *
     * @param[in] slot_id Slot id of the tensor in respect with the operator
     * @param[in] tensor  Tensor to be added in the pack
     *
     * @return Status code
     */
    StatusCode add(Tensor &tensor, int32_t slot_id)
    {
        return detail::as_enum<StatusCode>(AclPackTensor(_object.get(), tensor.get(), slot_id));
    }
    /** Add a list of tensors to a tensor pack
     *
     * @param[in] packed Pair packs to be added
     *
     * @return Status code
     */
    StatusCode add(std::initializer_list<PackPair> packed)
    {
        const size_t           size = packed.size();
        std::vector<int32_t>   slots(size);
        std::vector<AclTensor> tensors(size);
        int                    i = 0;
        for(auto &p : packed)
        {
            slots[i]   = p.slot_id;
            tensors[i] = AclTensor(p.tensor);
            ++i;
        }
        return detail::as_enum<StatusCode>(AclPackTensors(_object.get(), tensors.data(), slots.data(), size));
    }
};

/** Operator class
 *
 * Operators are the basic algorithmic blocks responsible for performing distinct operations
 */
class Operator : public detail::ObjectBase<AclOperator_>
{
public:
    /** Run an operator on a given input list
     *
     * @param[in,out] queue Queue to scheduler the operator on
     * @param pack  Tensor list to be used as input
     *
     * @return Status Code
     */
    StatusCode run(Queue &queue, TensorPack &pack)
    {
        return detail::as_cenum<StatusCode>(AclRunOperator(_object.get(), queue.get(), pack.get()));
    }

protected:
    /** Constructor */
    Operator() = default;
};

/// Operators
using ActivationDesc = AclActivationDescriptor;
class Activation : public Operator
{
public:
    Activation(Context &ctx, const TensorDescriptor &src, const TensorDescriptor &dst, const ActivationDesc &desc, StatusCode *status = nullptr)
    {
        AclOperator op;
        const auto  st = detail::as_enum<StatusCode>(AclActivation(&op, ctx.get(), src.get(), dst.get(), desc));
        reset(op);
        report_status(st, "[Compute Library] Failure during Activation operator creation");
        if(status)
        {
            *status = st;
        }
    }
};
} // namespace acl
#undef ARM_COMPUTE_IGNORE_UNUSED
#endif /* ARM_COMPUTE_ACL_HPP_ */
