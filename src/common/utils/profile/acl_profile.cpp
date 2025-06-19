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
#if defined(ACL_PROFILE_ENABLE) && (ACL_PROFILE_BACKEND == PERFETTO)
#include "src/common/utils/profile/acl_profile.h"

#include <fstream>

PERFETTO_TRACK_EVENT_STATIC_STORAGE();
namespace arm_compute
{
namespace profile
{

PerfettoProfiler::PerfettoProfiler()
    : tracing_session(nullptr),
      trace_start_ns(perfetto::TrackEvent::GetTraceTimeNs())
#ifdef ARM_COMPUTE_CL
      ,
      opencl_clock(nullptr),
      opencl_tracing_enabled(false)
#else
#endif
{
    perfetto::TracingInitArgs args;
    args.backends = perfetto::ACL_PROFILE_MODE;
    perfetto::Tracing::Initialize(args);
    perfetto::TrackEvent::Register();
    perfetto::TraceConfig cfg;
    cfg.add_buffers()->set_size_kb(ACL_ACL_PROFILE_SIZE_KB);
    cfg.add_data_sources()->mutable_config()->set_name("track_event");
    tracing_session = perfetto::Tracing::NewTrace();
    tracing_session->Setup(cfg);
    tracing_session->StartBlocking();
}

PerfettoProfiler::~PerfettoProfiler()
{
    if (tracing_session)
    {
        perfetto::TrackEvent::Flush();
        tracing_session->StopBlocking();
        auto          data = tracing_session->ReadTraceBlocking();
        std::ofstream out("acl.pftrace", std::ios::binary);
        out.write(data.data(), data.size());
        out.close();
    }
}

#ifdef ARM_COMPUTE_CL
void PerfettoProfiler::openclTraceBegin()
{
    // Lock the process to ensure that the tracing session is created only once
    if (!opencl_tracing_enabled && ACL_PROFILE_LEVEL > 0)
    {
#ifdef ARM_COMPUTE_CL
        opencl_clock = std::make_unique<OpenCLClock<true>>(ScaleFactor::NONE);
        if (!opencl_clock)
        {
            std::cerr << "Failed to create OpenCLClock instance." << std::endl;
        }
#endif
        opencl_clock->test_start();
        opencl_clock->start();
        opencl_tracing_enabled = true;
    }
}

void PerfettoProfiler::openclTraceEnd()
{
    uint64_t cpu_sync_time = getTsNs();
    ARM_COMPUTE_TRACE_CUSTOM_EVENT_END(ARM_COMPUTE_PROF_CAT_SCHEDULER, 0, cpu_sync_time);
    if (!opencl_clock || !opencl_tracing_enabled || ACL_PROFILE_LEVEL < 1)
    {
        return;
    }
    opencl_clock->stop();
    opencl_clock->test_stop();

#if (ACL_PROFILE_LEVEL > 1)
    // Print the RAW GPU timestamps
    std::cout << "RAW GPU timestamps:" << std::endl;
    for (const auto &instrument : opencl_clock->measurements())
    {
        std::cout << instrument.first << ": " << instrument.second << std::endl;
    }
#endif

    // The difference between the instrument map and this map is that.
    // MeasurementsMap elements does have an awareness of the timestamps in other GPU stages.
    // Gathering all the timestamps stages in the value of the map makes drawing spans in the CPU timeline easier.
    // |-------------------+----------------------------------------------------------------|
    // | Map Name          | Key                      | Value                               |
    // |-------------------+--------------------------+-------------------------------------|
    // | MeasurementsMap   | [stage][kernel]#ID       | GPU timestamp as string             |
    // |                   | (e.g., "[start]foo#1" )  | (e.g., "123456789 ns")              |
    // |-------------------+--------------------------+-------------------------------------|
    // | gpu_spans_map     | [kernel]#ID              | vector of stage timestamps (CPU ns) |
    // |                   | (e.g., "foo#1" )         | [queued, flushed, start, end]       |
    // |-------------------+--------------------------+-------------------------------------|
    std::map<std::string, std::vector<uint64_t>> gpu_spans_map;

    uint64_t gpu_sync_time = 0;
    // TODO : find a better way to sync GPU and CPU times
    // Here we are finding the GPU timestamp that have the highest value.
    // This is the closest to the end of ::sync() call.
    for (const auto &instrument : opencl_clock->measurements())
    {
        uint64_t gpu_ts = std::stoull(instrument.second.value().to_string());
        if (gpu_ts > gpu_sync_time)
        {
            gpu_sync_time = gpu_ts;
        }
    }

    for (const auto &instrument : opencl_clock->measurements())
    {
        const std::string &key      = instrument.first;
        const std::string  time_str = instrument.second.value().to_string();
        uint64_t           gpu_time = std::stoull(time_str);
        uint64_t           cpu_time = gpu_time + cpu_sync_time - gpu_sync_time;

        if (key.empty() || key[0] != '[')
            continue;

        // Find the closing bracket
        size_t end_bracket = key.find(']');
        if (end_bracket == std::string::npos)
            continue;

        std::string stage  = key.substr(1, end_bracket - 1);
        std::string kernel = key.substr(end_bracket + 1);

        int index = -1;
        if (stage == "queued")
            index = 0;
        else if (stage == "flushed")
            index = 1;
        else if (stage == "start")
            index = 2;
        else if (stage == "end")
            index = 3;
        else
            continue;

        auto &vec = gpu_spans_map[kernel];
        if (vec.size() < 4)
            vec.resize(4, 0);
        vec[index] = cpu_time;
    }

    for (auto &instrument : gpu_spans_map)
    {
#if (ACL_PROFILE_LEVEL > 1)
        std::cout << "Kernel: " << instrument.first << std::endl;
        std::cout << "Queued: " << instrument.second[0] << " ns" << std::endl;
        std::cout << "Flushed: " << instrument.second[1] << " ns" << std::endl;
        std::cout << "Start: " << instrument.second[2] << " ns" << std::endl;
        std::cout << "End: " << instrument.second[3] << " ns" << std::endl;
        std::cout << std::endl;
#endif

        ARM_COMPUTE_TRACE_CUSTOM_EVENT(ARM_COMPUTE_PROF_CAT_GPU, ARM_COMPUTE_PROF_LVL_GPU, instrument.second[0],
                                       instrument.second[1] - instrument.second[0], "GPU::Queue",
                                       instrument.first.c_str());
        ARM_COMPUTE_TRACE_CUSTOM_EVENT(ARM_COMPUTE_PROF_CAT_GPU, ARM_COMPUTE_PROF_LVL_GPU, instrument.second[1],
                                       instrument.second[2] - instrument.second[1], "GPU::Flush",
                                       instrument.first.c_str());
        ARM_COMPUTE_TRACE_CUSTOM_EVENT(ARM_COMPUTE_PROF_CAT_GPU, ARM_COMPUTE_PROF_LVL_GPU, instrument.second[2],
                                       instrument.second[3] - instrument.second[2], "GPU::Run",
                                       instrument.first.c_str());
    }
    opencl_clock.reset();
    opencl_tracing_enabled = false;
}
#endif
uint64_t PerfettoProfiler::getTsNs() const
{
    return perfetto::TrackEvent::GetTraceTimeNs() - trace_start_ns;
}

static PerfettoProfiler acl_perfetto;

PerfettoProfiler &get_profiler()
{
    return acl_perfetto;
}

} // namespace profile
} // namespace arm_compute

#endif
