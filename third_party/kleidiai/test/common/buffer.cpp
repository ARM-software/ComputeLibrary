//
// SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "buffer.hpp"

#if defined(__linux__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#endif  // defined(__linux__) || defined(__APPLE__)

#include <algorithm>
#include <cstddef>
#include <functional>
#include <sstream>
#include <string>

#include "kai/kai_common.h"

namespace kai::test {

Buffer::Buffer(const size_t size) : Buffer(size, 0) {
}

Buffer::Buffer(const size_t size, const uint8_t init_value = 0) : m_user_buffer_size(size) {
    KAI_ASSUME_ALWAYS_MSG(size > 0, "Buffers must be of non-zero size");

    const char* val = getenv(buffer_policy_env_name);
    const std::string buffer_policy = (val != nullptr) ? std::string(val) : std::string("NONE");

    std::ostringstream oss;

    if (buffer_policy == "PROTECT_UNDERFLOW" || buffer_policy == "PROTECT_OVERFLOW") {
#if defined(__linux__) || defined(__APPLE__)
        m_protection_policy = (buffer_policy == "PROTECT_UNDERFLOW") ? BufferProtectionPolicy::ProtectUnderflow
                                                                     : BufferProtectionPolicy::ProtectOverflow;
#else   // defined(__linux__) || defined(__APPLE__)
        oss << buffer_policy << " buffer protection policy is not supported on target platform";
#endif  // defined(__linux__) || defined(__APPLE__)
    } else if (buffer_policy == "NONE") {
        m_protection_policy = BufferProtectionPolicy::None;
    } else {
        oss << "Unrecognized buffer protection policy provided by " << buffer_policy_env_name << ": ";
        oss << buffer_policy;
    }

    if (!oss.str().empty()) {
        KAI_ERROR(oss.str().c_str());
    }

    switch (m_protection_policy) {
#if defined(__linux__) || defined(__APPLE__)
        case BufferProtectionPolicy::ProtectUnderflow:
        case BufferProtectionPolicy::ProtectOverflow:
            allocate_with_guard_pages();
            break;
#endif  // defined(__linux__) || defined(__APPLE__)
        default:
            allocate();
    }

    memset(data(), init_value, size);
}

void Buffer::allocate() {
    m_buffer = handle(std::malloc(m_user_buffer_size), &std::free);
    KAI_ASSUME_ALWAYS_MSG(m_buffer.get() != nullptr, "Failure allocating memory");
    KAI_ASSUME_ALWAYS_MSG(m_user_buffer_offset == 0, "Buffer offset must be zero for naive allocation");
}

#if defined(__linux__) || defined(__APPLE__)
void Buffer::allocate_with_guard_pages() {
    const auto sc_pagesize_res = sysconf(_SC_PAGESIZE);
    KAI_ASSUME_ALWAYS_MSG(sc_pagesize_res != -1, "Error finding page size");

    const auto page_size = static_cast<size_t>(sc_pagesize_res);

    // Offset the user buffer by the size of the first guard page
    m_user_buffer_offset = page_size;

    // The user buffer is rounded to the size of the nearest whole page.
    // This forms the valid region between the two guard pages
    const size_t valid_region_size = kai_roundup(m_user_buffer_size, page_size);
    const size_t protected_region_size = 2 * page_size;
    const size_t total_memory_size = valid_region_size + protected_region_size;

    if (m_protection_policy == BufferProtectionPolicy::ProtectOverflow) {
        // To detect overflows we offset the user buffer so that edge of the buffer is aligned to the start of the
        // higher guard page thus detecting whenever a buffer overflow occurs.
        m_user_buffer_offset += valid_region_size - m_user_buffer_size;
    }

    auto mmap_deleter = [total_memory_size](void* ptr) {
        if (munmap(ptr, total_memory_size) != 0) {
            KAI_ERROR("Failure deleting memory mappings");
        }
    };

    m_buffer =
        handle(mmap(nullptr, total_memory_size, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0), mmap_deleter);
    if (m_buffer.get() == MAP_FAILED) {
        KAI_ERROR("Failure mapping memory");
    }

    void* head_guard_page = m_buffer.get();
    void* tail_guard_page = static_cast<std::byte*>(m_buffer.get()) + (total_memory_size - page_size);

    if (mprotect(head_guard_page, std::max(static_cast<size_t>(0), page_size), PROT_NONE) != 0) {
        KAI_ERROR("Failure protecting page immediately preceding buffer");
    }
    if (mprotect(tail_guard_page, std::max(static_cast<size_t>(0), page_size), PROT_NONE) != 0) {
        KAI_ERROR("Failure protecting page immediately following buffer");
    }
}
#endif  // defined(__linux__) || defined(__APPLE__)

}  // namespace kai::test
