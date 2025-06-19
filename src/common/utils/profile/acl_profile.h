/*
 * Copyright (c) 2025 Arm Limited.
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

#ifndef ACL_SRC_COMMON_UTILS_PROFILE_ACL_PROFILE_H
#define ACL_SRC_COMMON_UTILS_PROFILE_ACL_PROFILE_H

// Define ACL profile categories
#define ARM_COMPUTE_PROF_CAT_NONE      "NONE"
#define ARM_COMPUTE_PROF_CAT_CPU       "CPU"
#define ARM_COMPUTE_PROF_CAT_NEON      "NEON"
#define ARM_COMPUTE_PROF_CAT_SVE       "SVE"
#define ARM_COMPUTE_PROF_CAT_SME       "SME"
#define ARM_COMPUTE_PROF_CAT_GPU       "GPU"
#define ARM_COMPUTE_PROF_CAT_MEMORY    "MEMORY"
#define ARM_COMPUTE_PROF_CAT_RUNTIME   "RUNTIME"
#define ARM_COMPUTE_PROF_CAT_SCHEDULER "SCHEDULER"

// Define ACL profile levels
#define ARM_COMPUTE_PROF_LVL_CPU 0
#define ARM_COMPUTE_PROF_LVL_GPU 1

#if defined(ACL_PROFILE_ENABLE) && (ACL_PROFILE_BACKEND == PERFETTO)
#include "third_party/perfetto/perfetto.h"

#ifdef ARM_COMPUTE_CL
#include "tests/framework/instruments/OpenCLTimer.h"
using namespace arm_compute::test::framework;
#endif

PERFETTO_DEFINE_CATEGORIES(
    perfetto::Category(ARM_COMPUTE_PROF_CAT_NONE).SetTags("verbose").SetDescription("No Category"),
    perfetto::Category(ARM_COMPUTE_PROF_CAT_CPU).SetTags("verbose").SetDescription("ACL CPU tracing"),
    perfetto::Category(ARM_COMPUTE_PROF_CAT_NEON).SetTags("verbose").SetDescription("ACL NEON tracing"),
    perfetto::Category(ARM_COMPUTE_PROF_CAT_SVE).SetTags("verbose").SetDescription("ACL SVE tracing"),
    perfetto::Category(ARM_COMPUTE_PROF_CAT_SME).SetTags("verbose").SetDescription("ACL SME tracing"),
    perfetto::Category(ARM_COMPUTE_PROF_CAT_GPU).SetTags("verbose").SetDescription("ACL GPU tracing"),
    perfetto::Category(ARM_COMPUTE_PROF_CAT_MEMORY).SetTags("verbose").SetDescription("ACL Memory tracing"),
    perfetto::Category(ARM_COMPUTE_PROF_CAT_RUNTIME).SetTags("verbose").SetDescription("ACL Runtime tracing"),
    perfetto::Category(ARM_COMPUTE_PROF_CAT_SCHEDULER).SetTags("verbose").SetDescription("ACL Scheduler tracing"));

namespace arm_compute
{
namespace profile
{

class PerfettoProfiler final
{
public:
    PerfettoProfiler();
    ~PerfettoProfiler();

    uint64_t getTsNs() const;
#ifdef ARM_COMPUTE_CL
    void openclTraceBegin();
    void openclTraceEnd();
#endif

private:
    std::unique_ptr<perfetto::TracingSession> tracing_session;
    uint64_t                                  trace_start_ns;
#ifdef ARM_COMPUTE_CL
    std::unique_ptr<OpenCLClock<true>> opencl_clock;
    bool                               opencl_tracing_enabled;
#endif
};

// 👇 Singleton accessor declaration
PerfettoProfiler &get_profiler();

class ScopedPerfettoTrace final
{
public:
    ScopedPerfettoTrace(uint64_t start_ts) : _start_ts(start_ts)
    {
        // TODO : add support for categories
        (void)_start_ts;
    }

    ~ScopedPerfettoTrace()
    {
        uint64_t end_ts = get_profiler().getTsNs();
        if (end_ts < _start_ts)
            end_ts = _start_ts + 1;

        TRACE_EVENT_END(ARM_COMPUTE_PROF_CAT_NONE, end_ts);
    }

private:
    uint64_t _start_ts;
};

} // namespace profile
} // namespace arm_compute

#ifdef ARM_COMPUTE_CL
#define ARM_COMPUTE_TRACE_OPENCL_BEGIN() arm_compute::profile::get_profiler().openclTraceBegin()
#define ARM_COMPUTE_TRACE_OPENCL_SYNC()  arm_compute::profile::get_profiler().openclTraceEnd()
#else
#define ARM_COMPUTE_TRACE_OPENCL_BEGIN() (void)0
#define ARM_COMPUTE_TRACE_OPENCL_SYNC()  (void)0
#endif

// This is useful to postprocess and recreate spans that did not happen on real time.
// Typical use of this is to redraw GPU spans in the CPU timeline view.
// Once we collect the GPU timestamps, we can use them to create spans in the CPU timeline.
#if ACL_PROFILE_LEVEL >= 1
#define USE_CUSTOM_TIMESTAMP
#endif

#define ARM_COMPUTE_TRACE_CUSTOM_EVENT(category, level, timestamp_ns, duration_ns, name, arg)                 \
    do                                                                                                        \
    {                                                                                                         \
        if ((int)(level) <= ACL_PROFILE_LEVEL)                                                                \
        {                                                                                                     \
            TRACE_EVENT_BEGIN(category, name, perfetto::Track(0xFF), (uint64_t)timestamp_ns, "arg", arg);     \
            TRACE_EVENT_END(category, perfetto::Track(0xFF), (uint64_t)timestamp_ns + (uint64_t)duration_ns); \
        }                                                                                                     \
    } while (0)

#define ARM_COMPUTE_TRACE_CUSTOM_EVENT_BEGIN(category, level, timestamp_ns, name, arg) \
    do                                                                                 \
    {                                                                                  \
        if ((int)(level) <= ACL_PROFILE_LEVEL)                                         \
        {                                                                              \
            TRACE_EVENT_BEGIN(category, name, (uint64_t)timestamp_ns);                 \
        }                                                                              \
    } while (0)

#define ARM_COMPUTE_TRACE_CUSTOM_EVENT_END(category, level, timestamp_ns) \
    do                                                                    \
    {                                                                     \
        if ((int)(level) <= ACL_PROFILE_LEVEL)                            \
            TRACE_EVENT_END(category, (uint64_t)timestamp_ns);            \
    } while (0)

#define ARM_COMPUTE_TRACE_CUSTOM_EVENT_INSTANT(category, level, name, timestamp_ns) \
    do                                                                              \
    {                                                                               \
        if ((int)(level) <= ACL_PROFILE_LEVEL)                                      \
            TRACE_EVENT_INSTANT(category, name, (uint64_t)timestamp_ns);            \
    } while (0)

#define ARM_COMPUTE_TRACE_EVENT(...) _ARM_COMPUTE_TRACE_EVENT_SELECT(__VA_ARGS__)

#define _ARM_COMPUTE_TRACE_EVENT_SELECT(cat, level, name) _ARM_COMPUTE_TRACE_EVENT_L##level(cat, name)

#ifdef USE_CUSTOM_TIMESTAMP

// TODO : find a better way to handle category with scoped trace event.
// At the moment these event category are set to None.
#if ACL_PROFILE_LEVEL >= 0
#define _ARM_COMPUTE_TRACE_EVENT_L0(category, name)                               \
    uint64_t __ts_##__COUNTER__ = arm_compute::profile::get_profiler().getTsNs(); \
    TRACE_EVENT_BEGIN(ARM_COMPUTE_PROF_CAT_NONE, name, __ts_##__COUNTER__);       \
    arm_compute::profile::ScopedPerfettoTrace __trace_scope_##__COUNTER__(__ts_##__COUNTER__);
#endif
#define ARM_COMPUTE_TRACE_EVENT_BEGIN(category, level, name)                                             \
    do                                                                                                   \
    {                                                                                                    \
        if ((int)(level) <= ACL_PROFILE_LEVEL)                                                           \
            TRACE_EVENT_BEGIN(category, name, (uint64_t)arm_compute::profile::get_profiler().getTsNs()); \
    } while (0)

#define ARM_COMPUTE_TRACE_EVENT_END(category, level)                                             \
    do                                                                                           \
    {                                                                                            \
        if ((int)(level) <= ACL_PROFILE_LEVEL)                                                   \
            TRACE_EVENT_END(category, (uint64_t)arm_compute::profile::get_profiler().getTsNs()); \
    } while (0)

#define ARM_COMPUTE_TRACE_EVENT_INSTANT(category, level, name)                                             \
    do                                                                                                     \
    {                                                                                                      \
        if ((int)(level) <= ACL_PROFILE_LEVEL)                                                             \
            TRACE_EVENT_INSTANT(category, name, (uint64_t)arm_compute::profile::get_profiler().getTsNs()); \
    } while (0)

#define ARM_COMPUTE_TRACE_COUNTER(category, level, name, value)                                             \
    do                                                                                                      \
    {                                                                                                       \
        if ((int)(level) <= ACL_PROFILE_LEVEL)                                                              \
            TRACE_COUNTER(category, name, (uint64_t)arm_compute::profile::get_profiler().getTsNs(), value); \
    } while (0)
#else

#if ACL_PROFILE_LEVEL >= 0
#define _ARM_COMPUTE_TRACE_EVENT_L0(category, name) TRACE_EVENT(category, name)
#endif

#define ARM_COMPUTE_TRACE_EVENT_BEGIN(category, level, name) \
    do                                                       \
    {                                                        \
        if ((int)(level) <= ACL_PROFILE_LEVEL)               \
            TRACE_EVENT_BEGIN(category, name);               \
    } while (0)

#define ARM_COMPUTE_TRACE_EVENT_END(category, level) \
    do                                               \
    {                                                \
        if ((int)(level) <= ACL_PROFILE_LEVEL)       \
            TRACE_EVENT_END(category);               \
    } while (0)

#define ARM_COMPUTE_TRACE_EVENT_INSTANT(category, level, name) \
    do                                                         \
    {                                                          \
        if ((int)(level) <= ACL_PROFILE_LEVEL)                 \
            TRACE_EVENT_INSTANT(category, name);               \
    } while (0)

#define ARM_COMPUTE_TRACE_COUNTER(category, level, name, value) \
    do                                                          \
    {                                                           \
        if ((int)(level) <= ACL_PROFILE_LEVEL)                  \
            TRACE_COUNTER(category, name, value);               \
    } while (0)
#endif

#if ACL_PROFILE_LEVEL >= 1
#define _ARM_COMPUTE_TRACE_EVENT_L1(category, name) _ARM_COMPUTE_TRACE_EVENT_L0(category, name)
#else
#define _ARM_COMPUTE_TRACE_EVENT_L1(category, name)
#endif

#if ACL_PROFILE_LEVEL >= 2
#define _ARM_COMPUTE_TRACE_EVENT_L2(category, name) _ARM_COMPUTE_TRACE_EVENT_L1(category, name)
#else
#define _ARM_COMPUTE_TRACE_EVENT_L2(category, name)
#endif

#else
// Stub PROFILE macros to do nothing
#define ARM_COMPUTE_TRACE_OPENCL_BEGIN() (void)0
#define ARM_COMPUTE_TRACE_OPENCL_SYNC()  (void)0
#define ARM_COMPUTE_TRACE_CUSTOM_EVENT_BEGIN(category, level, timestamp_ns, name, arg) \
    (void)category;                                                                    \
    (void)name;                                                                        \
    (void)level;                                                                       \
    (void)timestamp_ns(void) arg;
#define ARM_COMPUTE_TRACE_CUSTOM_EVENT_END(category, level, timestamp_ns) \
    (void)category;                                                       \
    (void)level;                                                          \
    (void)timestamp_ns
#define ARM_COMPUTE_TRACE_EVENT(category, level, name) \
    (void)category;                                    \
    (void)name;                                        \
    (void)level
#define ARM_COMPUTE_TRACE_EVENT_BEGIN(category, level, name) \
    (void)category;                                          \
    (void)name;                                              \
    (void)level
#define ARM_COMPUTE_TRACE_EVENT_END(category, level) \
    (void)category;                                  \
    (void)level
#define ARM_COMPUTE_TRACE_EVENT_INSTANT(category, level, name) \
    (void)category;                                            \
    (void)name;                                                \
    (void)level
#define ARM_COMPUTE_PROFILE_INIT() \
    do                             \
    {                              \
    } while (0)
#define ARM_COMPUTE_PROFILE_STATIC_STORAGE() \
    do                                       \
    {                                        \
    } while (0)
#define ARM_COMPUTE_PROFILE_FINISH() \
    do                               \
    {                                \
    } while (0)
#endif // ACL_PROFILE_ENABLE

#endif // ACL_SRC_COMMON_UTILS_PROFILE_ACL_PROFILE_H
