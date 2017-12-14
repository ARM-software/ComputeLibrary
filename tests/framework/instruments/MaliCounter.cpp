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
#include "MaliCounter.h"

#include "arm_compute/core/Error.h"

namespace arm_compute
{
namespace test
{
namespace framework
{
namespace
{
struct MaliHWInfo
{
    unsigned mp_count;
    unsigned gpu_id;
    unsigned r_value;
    unsigned p_value;
    unsigned core_mask;
};

MaliHWInfo get_mali_hw_info(const char *path)
{
    int fd = open(path, O_RDWR); // NOLINT

    if(fd < 0)
    {
        ARM_COMPUTE_ERROR("Failed to get HW info.");
    }

    {
        mali_userspace::uku_version_check_args version_check_args;                // NOLINT
        version_check_args.header.id = mali_userspace::UKP_FUNC_ID_CHECK_VERSION; // NOLINT
        version_check_args.major     = 10;
        version_check_args.minor     = 2;

        if(mali_userspace::mali_ioctl(fd, version_check_args) != 0)
        {
            ARM_COMPUTE_ERROR("Failed to check version.");
            close(fd);
        }
    }

    {
        mali_userspace::kbase_uk_hwcnt_reader_set_flags flags; // NOLINT
        memset(&flags, 0, sizeof(flags));
        flags.header.id    = mali_userspace::KBASE_FUNC_SET_FLAGS; // NOLINT
        flags.create_flags = mali_userspace::BASE_CONTEXT_CREATE_KERNEL_FLAGS;

        if(mali_userspace::mali_ioctl(fd, flags) != 0)
        {
            ARM_COMPUTE_ERROR("Failed settings flags ioctl.");
            close(fd);
        }
    }

    {
        mali_userspace::kbase_uk_gpuprops props;                         // NOLINT
        props.header.id = mali_userspace::KBASE_FUNC_GPU_PROPS_REG_DUMP; // NOLINT

        if(mali_ioctl(fd, props) != 0)
        {
            ARM_COMPUTE_ERROR("Failed settings flags ioctl.");
            close(fd);
        }

        MaliHWInfo hw_info; // NOLINT
        memset(&hw_info, 0, sizeof(hw_info));
        hw_info.gpu_id  = props.props.core_props.product_id;
        hw_info.r_value = props.props.core_props.major_revision;
        hw_info.p_value = props.props.core_props.minor_revision;

        for(unsigned int i = 0; i < props.props.coherency_info.num_core_groups; ++i)
        {
            hw_info.core_mask |= props.props.coherency_info.group[i].core_mask;
        }

        hw_info.mp_count = __builtin_popcountll(hw_info.core_mask);

        close(fd);

        return hw_info;
    }
}
} // namespace

MaliCounter::MaliCounter(ScaleFactor scale_factor)
{
    _counters =
    {
        { "GPU_ACTIVE", Measurement(0, "cycles") },
    };

    _core_counters =
    {
        { "ARITH_WORDS", { "Arithmetic pipe", std::map<int, uint64_t>(), "instructions" } },
        { "LS_ISSUE", { "LS pipe", std::map<int, uint64_t>(), "instructions" } },
        { "TEX_ISSUE", { "Texture pipe", std::map<int, uint64_t>(), "instructions" } },
        { "COMPUTE_ACTIVE", { "Compute core", std::map<int, uint64_t>(), "cycles" } },
        { "FRAG_ACTIVE", { "Fragment core", std::map<int, uint64_t>(), "cycles" } },
    };

    switch(scale_factor)
    {
        case ScaleFactor::NONE:
            _scale_factor = 1;
            _unit         = "";
            break;
        case ScaleFactor::SCALE_1K:
            _scale_factor = 1000;
            _unit         = "K ";
            break;
        case ScaleFactor::SCALE_1M:
            _scale_factor = 1000000;
            _unit         = "M ";
            break;
        default:
            ARM_COMPUTE_ERROR("Invalid scale");
    }

    init();
}

MaliCounter::~MaliCounter()
{
    term();
}

void MaliCounter::init()
{
    term();

    MaliHWInfo hw_info = get_mali_hw_info(_device);

    _num_cores = hw_info.mp_count;

    _fd = open(_device, O_RDWR | O_CLOEXEC | O_NONBLOCK); // NOLINT

    if(_fd < 0)
    {
        ARM_COMPUTE_ERROR("Failed to open /dev/mali0.");
    }

    {
        mali_userspace::kbase_uk_hwcnt_reader_version_check_args check; // NOLINT
        memset(&check, 0, sizeof(check));

        if(mali_userspace::mali_ioctl(_fd, check) != 0)
        {
            ARM_COMPUTE_ERROR("Failed to get ABI version.");
        }
        else if(check.major < 10)
        {
            ARM_COMPUTE_ERROR("Unsupported ABI version 10.");
        }
    }

    {
        mali_userspace::kbase_uk_hwcnt_reader_set_flags flags; // NOLINT
        memset(&flags, 0, sizeof(flags));
        flags.header.id    = mali_userspace::KBASE_FUNC_SET_FLAGS; // NOLINT
        flags.create_flags = mali_userspace::BASE_CONTEXT_CREATE_KERNEL_FLAGS;

        if(mali_userspace::mali_ioctl(_fd, flags) != 0)
        {
            ARM_COMPUTE_ERROR("Failed settings flags ioctl.");
        }
    }

    {
        mali_userspace::kbase_uk_hwcnt_reader_setup setup; // NOLINT
        memset(&setup, 0, sizeof(setup));
        setup.header.id    = mali_userspace::KBASE_FUNC_HWCNT_READER_SETUP; // NOLINT
        setup.buffer_count = _buffer_count;
        setup.jm_bm        = -1;
        setup.shader_bm    = -1;
        setup.tiler_bm     = -1;
        setup.mmu_l2_bm    = -1;
        setup.fd           = -1;

        if(mali_userspace::mali_ioctl(_fd, setup) != 0)
        {
            ARM_COMPUTE_ERROR("Failed setting hwcnt reader ioctl.");
        }

        _hwc_fd = setup.fd;
    }

    {
        uint32_t api_version = ~mali_userspace::HWCNT_READER_API;

        if(ioctl(_hwc_fd, mali_userspace::KBASE_HWCNT_READER_GET_API_VERSION, &api_version) != 0) // NOLINT
        {
            ARM_COMPUTE_ERROR("Could not determine hwcnt reader API.");
        }
        else if(api_version != mali_userspace::HWCNT_READER_API)
        {
            ARM_COMPUTE_ERROR("Invalid API version.");
        }
    }

    if(ioctl(_hwc_fd, static_cast<int>(mali_userspace::KBASE_HWCNT_READER_GET_BUFFER_SIZE), &_buffer_size) != 0) // NOLINT
    {
        ARM_COMPUTE_ERROR("Failed to get buffer size.");
    }

    if(ioctl(_hwc_fd, static_cast<int>(mali_userspace::KBASE_HWCNT_READER_GET_HWVER), &_hw_ver) != 0) // NOLINT
    {
        ARM_COMPUTE_ERROR("Could not determine HW version.");
    }

    if(_hw_ver < 5)
    {
        ARM_COMPUTE_ERROR("Unsupported HW version.");
    }

    _sample_data = static_cast<uint8_t *>(mmap(nullptr, _buffer_count * _buffer_size, PROT_READ, MAP_PRIVATE, _hwc_fd, 0));

    if(_sample_data == MAP_FAILED) // NOLINT
    {
        ARM_COMPUTE_ERROR("Failed to map sample data.");
    }

    auto product = std::find_if(std::begin(mali_userspace::products), std::end(mali_userspace::products), [&](const mali_userspace::CounterMapping & cm)
    {
        return (cm.product_mask & hw_info.gpu_id) == cm.product_id;
    });

    if(product != std::end(mali_userspace::products))
    {
        _names_lut = product->names_lut;
    }
    else
    {
        ARM_COMPUTE_ERROR("Could not identify GPU.");
    }

    _raw_counter_buffer.resize(_buffer_size / sizeof(uint32_t));

    // Build core remap table.
    _core_index_remap.clear();
    _core_index_remap.reserve(hw_info.mp_count);

    unsigned int mask = hw_info.core_mask;

    while(mask != 0)
    {
        unsigned int bit = __builtin_ctz(mask);
        _core_index_remap.push_back(bit);
        mask &= ~(1u << bit);
    }
}

void MaliCounter::term()
{
    if(_sample_data != nullptr)
    {
        munmap(_sample_data, _buffer_count * _buffer_size);
        _sample_data = nullptr;
    }

    if(_hwc_fd >= 0)
    {
        close(_hwc_fd);
        _hwc_fd = -1;
    }

    if(_fd >= 0)
    {
        close(_fd);
        _fd = -1;
    }
}

void MaliCounter::sample_counters()
{
    if(ioctl(_hwc_fd, mali_userspace::KBASE_HWCNT_READER_DUMP, 0) != 0)
    {
        ARM_COMPUTE_ERROR("Could not sample hardware counters.");
    }
}

void MaliCounter::wait_next_event()
{
    pollfd poll_fd; // NOLINT
    poll_fd.fd     = _hwc_fd;
    poll_fd.events = POLLIN;

    const int count = poll(&poll_fd, 1, -1);

    if(count < 0)
    {
        ARM_COMPUTE_ERROR("poll() failed.");
    }

    if((poll_fd.revents & POLLIN) != 0)
    {
        mali_userspace::kbase_hwcnt_reader_metadata meta; // NOLINT

        if(ioctl(_hwc_fd, static_cast<int>(mali_userspace::KBASE_HWCNT_READER_GET_BUFFER), &meta) != 0) // NOLINT
        {
            ARM_COMPUTE_ERROR("Failed READER_GET_BUFFER.");
        }

        memcpy(_raw_counter_buffer.data(), _sample_data + _buffer_size * meta.buffer_idx, _buffer_size);
        _timestamp = meta.timestamp;

        if(ioctl(_hwc_fd, mali_userspace::KBASE_HWCNT_READER_PUT_BUFFER, &meta) != 0) // NOLINT
        {
            ARM_COMPUTE_ERROR("Failed READER_PUT_BUFFER.");
        }
    }
    else if((poll_fd.revents & POLLHUP) != 0)
    {
        ARM_COMPUTE_ERROR("HWC hung up.");
    }
}

const uint32_t *MaliCounter::get_counters() const
{
    return _raw_counter_buffer.data();
}

const uint32_t *MaliCounter::get_counters(mali_userspace::MaliCounterBlockName block, int core) const
{
    switch(block)
    {
        case mali_userspace::MALI_NAME_BLOCK_JM:
            return _raw_counter_buffer.data() + mali_userspace::MALI_NAME_BLOCK_SIZE * 0;
        case mali_userspace::MALI_NAME_BLOCK_MMU:
            return _raw_counter_buffer.data() + mali_userspace::MALI_NAME_BLOCK_SIZE * 2;
        case mali_userspace::MALI_NAME_BLOCK_TILER:
            return _raw_counter_buffer.data() + mali_userspace::MALI_NAME_BLOCK_SIZE * 1;
        default:
            if(core < 0)
            {
                ARM_COMPUTE_ERROR("Invalid core number.");
            }

            return _raw_counter_buffer.data() + mali_userspace::MALI_NAME_BLOCK_SIZE * (3 + _core_index_remap[core]);
    }
}

int MaliCounter::find_counter_index_by_name(mali_userspace::MaliCounterBlockName block, const char *name)
{
    const char *const *names = &_names_lut[mali_userspace::MALI_NAME_BLOCK_SIZE * block];

    for(int i = 0; i < mali_userspace::MALI_NAME_BLOCK_SIZE; ++i)
    {
        if(strstr(names[i], name) != nullptr)
        {
            return i;
        }
    }

    return -1;
}

void MaliCounter::start()
{
    sample_counters();
    wait_next_event();
    _start_time = _timestamp;
}

void MaliCounter::stop()
{
    sample_counters();
    wait_next_event();

    const uint32_t *counter    = get_counters(mali_userspace::MALI_NAME_BLOCK_JM);
    _counters.at("GPU_ACTIVE") = Measurement(counter[find_counter_index_by_name(mali_userspace::MALI_NAME_BLOCK_JM, "GPU_ACTIVE")], _counters.at("GPU_ACTIVE").unit());

    const int arith_index   = find_counter_index_by_name(mali_userspace::MALI_NAME_BLOCK_SHADER, "ARITH_WORDS");
    const int ls_index      = find_counter_index_by_name(mali_userspace::MALI_NAME_BLOCK_SHADER, "LS_ISSUE");
    const int tex_index     = find_counter_index_by_name(mali_userspace::MALI_NAME_BLOCK_SHADER, "TEX_ISSUE");
    const int compute_index = find_counter_index_by_name(mali_userspace::MALI_NAME_BLOCK_SHADER, "COMPUTE_ACTIVE");
    const int frag_index    = find_counter_index_by_name(mali_userspace::MALI_NAME_BLOCK_SHADER, "FRAG_ACTIVE");

    // Shader core counters can be averaged if desired, but here we don't.
    for(int core = 0; core < _num_cores; ++core)
    {
        const uint32_t *sc_counter = get_counters(mali_userspace::MALI_NAME_BLOCK_SHADER, core);

        _core_counters.at("ARITH_WORDS").values[core]    = sc_counter[arith_index];
        _core_counters.at("LS_ISSUE").values[core]       = sc_counter[ls_index];
        _core_counters.at("TEX_ISSUE").values[core]      = sc_counter[tex_index];
        _core_counters.at("COMPUTE_ACTIVE").values[core] = sc_counter[compute_index];
        _core_counters.at("FRAG_ACTIVE").values[core]    = sc_counter[frag_index];
    }

    _stop_time = _timestamp;
}

std::string MaliCounter::id() const
{
    return "Mali Counter";
}

Instrument::MeasurementsMap MaliCounter::measurements() const
{
    Measurement counters((_counters.at("GPU_ACTIVE").value() / _scale_factor).v.floating_point, _unit + _counters.at("GPU_ACTIVE").unit()); //NOLINT

    MeasurementsMap measurements
    {
        { "Timespan", Measurement(_stop_time - _start_time, "ns") },
        { "GPU active", counters },
    };

    for(const auto &counter : _core_counters)
    {
        for(const auto &core : counter.second.values)
        {
            measurements.emplace(counter.second.name + " #" + support::cpp11::to_string(core.first), Measurement(core.second / _scale_factor, _unit + counter.second.unit));
        }
    }

    return measurements;
}
} // namespace framework
} // namespace test
} // namespace arm_compute
