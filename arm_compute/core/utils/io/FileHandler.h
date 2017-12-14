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
#ifndef __ARM_COMPUTE_IO_FILE_HANDLER_H__
#define __ARM_COMPUTE_IO_FILE_HANDLER_H__

#include <fstream>
#include <string>

namespace arm_compute
{
namespace io
{
/** File Handling interface */
class FileHandler
{
public:
    /** Default Constructor */
    FileHandler();
    /** Default Destructor */
    ~FileHandler();
    /** Allow instances of this class to be moved */
    FileHandler(FileHandler &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    FileHandler(const FileHandler &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    FileHandler &operator=(const FileHandler &) = delete;
    /** Allow instances of this class to be moved */
    FileHandler &operator=(FileHandler &&) = default;
    /** Opens file
     *
     * @param[in] filename File name
     * @param[in] mode     File open mode
     */
    void open(const std::string &filename, std::ios_base::openmode mode);
    /** Closes file */
    void close();
    /** Returns the file stream
     *
     * @return File stream
     */
    std::fstream &stream();
    /** Returns filename of the handled file
     *
     * @return File filename
     */
    std::string filename() const;

private:
    std::fstream            _filestream;
    std::string             _filename;
    std::ios_base::openmode _mode;
};
} // namespace io
} // namespace arm_compute
#endif /* __ARM_COMPUTE_IO_FILE_HANDLER_H__ */
