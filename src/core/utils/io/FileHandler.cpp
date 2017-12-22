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
#include <string>

#include "arm_compute/core/utils/io/FileHandler.h"

#include "arm_compute/core/Error.h"
#include "support/ToolchainSupport.h"

using namespace arm_compute::io;

FileHandler::FileHandler()
    : _filestream(), _filename(" "), _mode()
{
}

FileHandler::~FileHandler()
{
    close();
}

void FileHandler::open(const std::string &filename, std::ios_base::openmode mode)
{
    close();
    ;
    _filestream.open(filename, mode);
    ARM_COMPUTE_ERROR_ON(!_filestream.good());
    _filename = filename;
    _mode     = mode;
}

void FileHandler::close()
{
    _filestream.close();
}

std::fstream &FileHandler::stream()
{
    return _filestream;
}

std::string FileHandler::filename() const
{
    return _filename;
}
