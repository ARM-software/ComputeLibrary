/*
 * Copyright (c) 2020-2021 Arm Limited.
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
#ifndef ARM_COMPUTE_TRACEPOINT_H
#define ARM_COMPUTE_TRACEPOINT_H

#include <string>
#include <type_traits>
#include <vector>

namespace arm_compute
{
#ifdef ARM_COMPUTE_TRACING_ENABLED
#define ARM_COMPUTE_CREATE_TRACEPOINT(...) TracePoint __tp(__VA_ARGS__)

/** Class used to dump configuration values in functions and kernels  */
class TracePoint final
{
public:
    /** Layer types */
    enum class Layer
    {
        CORE,
        RUNTIME
    };
    /** struct describing the arguments for a tracepoint */
    struct Args final
    {
        std::vector<std::string> args{};
    };
    /** Constructor
     *
     * @param[in] source     type of layer for the tracepoint
     * @param[in] class_name the name of the class creating the tracepoint
     * @param[in] object     a pointer to the actual object owning the tracepoint
     * @param[in] args       a struct describing all the arguments used in the call to the configure() method
     *
     */
    TracePoint(Layer source, const std::string &class_name, void *object, Args &&args);
    /** Destructor */
    ~TracePoint();

private:
    static int g_depth; /**< current depth */
    int        _depth;  /**< tracepoint depth */
};

/** Operator to write an argument to a @ref TracePoint
 *
 * @param[in] tp  Tracepoint to be used for writing
 * @param[in] arg Argument to be written in the tracepoint
 *
 * @return A referece to the updated tracepoint
 */
template <typename T>
TracePoint::Args &&operator<<(typename std::enable_if < !std::is_pointer<T>::value, TracePoint::Args >::type &&tp, const T &arg);
template <typename T>
TracePoint::Args &&operator<<(TracePoint::Args &&tp, const T *arg);

#define ARM_COMPUTE_CONST_REF_CLASS(type)                                 \
    template <>                                                           \
    TracePoint::Args &&operator<<(TracePoint::Args &&tp, const type &arg) \
    {                                                                     \
        ARM_COMPUTE_UNUSED(tp);                                           \
        tp.args.push_back(#type "(" + to_string(arg) + ")");              \
        return std::move(tp);                                             \
    }

#define ARM_COMPUTE_CONST_PTR_ADDRESS(type)                               \
    template <>                                                           \
    TracePoint::Args &&operator<<(TracePoint::Args &&tp, const type *arg) \
    {                                                                     \
        ARM_COMPUTE_UNUSED(tp);                                           \
        tp.args.push_back(#type "*(" + to_ptr_string(arg) + ")");         \
        return std::move(tp);                                             \
    }
#define ARM_COMPUTE_CONST_PTR_CLASS(type)                                 \
    template <>                                                           \
    TracePoint::Args &&operator<<(TracePoint::Args &&tp, const type *arg) \
    {                                                                     \
        ARM_COMPUTE_UNUSED(tp);                                           \
        if(arg)                                                           \
            tp.args.push_back(#type "(" + to_string(*arg) + ")");         \
        else                                                              \
            tp.args.push_back(#type "( nullptr )");                       \
        return std::move(tp);                                             \
    }

#define ARM_COMPUTE_CONST_REF_SIMPLE(type)                                   \
    template <>                                                              \
    TracePoint::Args &&operator<<(TracePoint::Args &&tp, const type &arg)    \
    {                                                                        \
        ARM_COMPUTE_UNUSED(tp);                                              \
        tp.args.push_back(#type "(" + support::cpp11::to_string(arg) + ")"); \
        return std::move(tp);                                                \
    }

#define ARM_COMPUTE_TRACE_TO_STRING(type)  \
    std::string to_string(const type &arg) \
    {                                      \
        ARM_COMPUTE_UNUSED(arg);           \
        return "";                         \
    }
#else /* ARM_COMPUTE_TRACING_ENABLED */
#define ARM_COMPUTE_CREATE_TRACEPOINT(...)
#define ARM_COMPUTE_CONST_REF_CLASS(type)
#define ARM_COMPUTE_CONST_PTR_ADDRESS(type)
#define ARM_COMPUTE_CONST_PTR_CLASS(type)
#define ARM_COMPUTE_CONST_REF_SIMPLE(type)
#define ARM_COMPUTE_TRACE_TO_STRING(type)
#endif /* ARM_COMPUTE_TRACING_ENABLED */
} //namespace arm_compute

#endif /* ARM_COMPUTE_TRACEPOINT_H */
