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
#ifndef __ARM_COMPUTE_MISC_MMAPPED_FILE_H__
#define __ARM_COMPUTE_MISC_MMAPPED_FILE_H__

#if !defined(BARE_METAL)

#include <string>
#include <utility>

namespace arm_compute
{
namespace utils
{
namespace mmap_io
{
/** Memory mapped file class */
class MMappedFile
{
public:
    /** Constructor */
    MMappedFile();
    /** Constructor
     *
     * @note file will be created if it doesn't exist.
     *
     * @param[in] filename File to be mapped, if doesn't exist will be created.
     * @param[in] size     Size of file to map
     * @param[in] offset   Offset to mapping point, should be multiple of page size
     */
    MMappedFile(std::string filename, size_t size, size_t offset);
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    MMappedFile(const MMappedFile &) = delete;
    /** Default move constructor */
    MMappedFile(MMappedFile &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    MMappedFile &operator=(const MMappedFile &) = delete;
    /** Default move assignment operator */
    MMappedFile &operator=(MMappedFile &&) = default;
    /** Destructor */
    ~MMappedFile();
    /** Opens and maps a file
     *
     * @note file will be created if it doesn't exist.
     *
     * @param[in] filename File to be mapped, if doesn't exist will be created.
     * @param[in] size     Size of file to map. If 0 all the file will be mapped.
     * @param[in] offset   Offset to mapping point, should be multiple of page size.
     *
     * @return True if operation was successful else false
     */
    bool map(const std::string &filename, size_t size, size_t offset);
    /** Unmaps and closes file */
    void release();
    /** Mapped data accessor
     *
     * @return Pointer to the mapped data, nullptr if not mapped
     */
    unsigned char *data();
    /** File size accessor
     *
     * @return Size of file
     */
    size_t file_size() const;
    /** Map size accessor
     *
     * @return Mapping size
     */
    size_t map_size() const;
    /** Checks if file mapped
     *
     * @return True if file is mapped else false
     */
    bool is_mapped() const;

private:
    std::string _filename;
    size_t      _file_size;
    size_t      _map_size;
    size_t      _map_offset;
    FILE       *_fp;
    void       *_data;
};
} // namespace mmap_io
} // namespace utils
} // namespace arm_compute
#endif // !defined(BARE_METAL)

#endif /* __ARM_COMPUTE_MISC_MMAPPED_FILE_H__ */
