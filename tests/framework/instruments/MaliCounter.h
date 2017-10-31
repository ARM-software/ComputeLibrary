/*
 * Copyright (c) 2017 ARM Limited.
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
#ifndef ARM_COMPUTE_TEST_MALI_COUNTER
#define ARM_COMPUTE_TEST_MALI_COUNTER

#include "Instrument.h"
#include "Measurement.h"
#include "hwc.hpp"

#include <map>
#include <vector>

namespace arm_compute
{
namespace test
{
namespace framework
{
/** Instrument implementation for mali hw counters. */
class MaliCounter : public Instrument
{
public:
    /** Default constructor. */
    MaliCounter(ScaleFactor scale_factor);

    MaliCounter(const MaliCounter &) = delete;
    MaliCounter &operator=(const MaliCounter &) = delete;

    /** Default destructor */
    ~MaliCounter();

    std::string     id() const override;
    void            start() override;
    void            stop() override;
    MeasurementsMap measurements() const override;

private:
    void init();
    void term();

    void            sample_counters();
    void            wait_next_event();
    const uint32_t *get_counters() const;
    const uint32_t *get_counters(mali_userspace::MaliCounterBlockName block, int core = -1) const;
    int find_counter_index_by_name(mali_userspace::MaliCounterBlockName block, const char *name);

    std::map<std::string, Measurement> _counters{};

    struct core_counters
    {
        std::string name;
        std::map<int, uint64_t> values;
        std::string unit;
    };

    std::map<std::string, core_counters> _core_counters{};
    uint64_t _start_time{ 0 };
    uint64_t _stop_time{ 0 };

    const char *const  _device
    { "/dev/mali0"
    };
    int                _num_cores{ 0 };
    uint32_t           _hw_ver{ 0 };
    int                _buffer_count{ 16 };
    size_t             _buffer_size{ 0 };
    uint8_t           *_sample_data{ nullptr };
    uint64_t           _timestamp{ 0 };
    const char *const *_names_lut
    {
        nullptr
    };
    std::vector<uint32_t>     _raw_counter_buffer{};
    std::vector<unsigned int> _core_index_remap{};
    int                       _fd{ -1 };
    int                       _hwc_fd{ -1 };
    int                       _scale_factor{};
};
} // namespace framework
} // namespace test
} // namespace arm_compute
#endif /* ARM_COMPUTE_TEST_MALI_COUNTER */
