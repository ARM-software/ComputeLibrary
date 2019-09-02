/*
 * Copyright (c) 2019 ARM Limited.
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
#if !defined(BARE_METAL)

#include "arm_compute/core/utils/misc/MMappedFile.h"

#include <cstdio>
#include <cstring>
#include <tuple>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace arm_compute
{
namespace utils
{
namespace mmap_io
{
namespace
{
/** File size accessor
 *
 * @param[in] filename File to extract its size
 *
 * @return A pair of size and status.
 */
std::pair<size_t, bool> get_file_size(const std::string &filename)
{
    struct stat st; // NOLINT
    memset(&st, 0, sizeof(struct stat));
    if(stat(filename.c_str(), &st) == 0)
    {
        return std::make_pair(st.st_size, true);
    }
    else
    {
        return std::make_pair(0, false);
    }
}

/** Get OS page size
 *
 * @return Page size
 */
size_t get_page_size()
{
    return sysconf(_SC_PAGESIZE);
}
} // namespace

MMappedFile::MMappedFile()
    : _filename(), _file_size(0), _map_size(0), _map_offset(0), _fp(nullptr), _data(nullptr)
{
}

MMappedFile::MMappedFile(std::string filename, size_t size, size_t offset)
    : _filename(std::move(filename)), _file_size(0), _map_size(size), _map_offset(offset), _fp(nullptr), _data(nullptr)
{
    map(_filename, _map_size, _map_offset);
}

MMappedFile::~MMappedFile()
{
    release();
}

bool MMappedFile::map(const std::string &filename, size_t size, size_t offset)
{
    // Check if file is mapped
    if(is_mapped())
    {
        return false;
    }

    // Open file
    _fp = fopen(filename.c_str(), "a+be");
    if(_fp == nullptr)
    {
        return false;
    }

    // Extract file descriptor
    int  fd     = fileno(_fp);
    bool status = fd >= 0;
    if(status)
    {
        // Get file size
        std::tie(_file_size, status) = get_file_size(_filename);

        if(status)
        {
            // Map all file from offset if map size is 0
            _map_size   = (size == 0) ? _file_size : size;
            _map_offset = offset;

            // Check offset mapping
            if((_map_offset > _file_size) || (_map_offset % get_page_size() != 0))
            {
                status = false;
            }
            else
            {
                // Truncate to file size
                if(_map_offset + _map_size > _file_size)
                {
                    _map_size = _file_size - _map_offset;
                }

                // Perform mapping
                _data = ::mmap(nullptr, _map_size, PROT_WRITE, MAP_SHARED, fd, _map_offset);
            }
        }
    }

    if(!status)
    {
        fclose(_fp);
    }

    return status;
}

void MMappedFile::release()
{
    // Unmap file
    if(_data != nullptr)
    {
        ::munmap(_data, _file_size);
        _data = nullptr;
    }

    // Close file
    if(_fp != nullptr)
    {
        fclose(_fp);
        _fp = nullptr;
    }

    // Clear variables
    _file_size  = 0;
    _map_size   = 0;
    _map_offset = 0;
}

unsigned char *MMappedFile::data()
{
    return static_cast<unsigned char *>(_data);
}

size_t MMappedFile::file_size() const
{
    return _file_size;
}

size_t MMappedFile::map_size() const
{
    return _map_size;
}

bool MMappedFile::is_mapped() const
{
    return _data != nullptr;
}
} // namespace mmap_io
} // namespace utils
} // namespace arm_compute
#endif // !defined(BARE_METAL)
