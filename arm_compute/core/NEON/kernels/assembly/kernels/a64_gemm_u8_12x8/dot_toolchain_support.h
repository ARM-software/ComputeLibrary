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

// Define a macro to assemble the UDOT instruction (in the absence of toolchain support)
#define _DECLARE_UDOT ".altmacro\n"\
    ".macro udot opd:req, opn:req, opm:req\n"\
    "local vd, vn, vm, h, l\n"\
    ".irp reg,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31\n"\
    ".ifeqs \"\\opd\",\"v\\reg\\.4s\"\n"\
    ".set vd,\\reg\n"\
    ".endif\n"\
    ".ifeqs \"\\opn\",\"v\\reg\\.16b\"\n"\
    ".set vn,\\reg\n"\
    ".endif\n"\
    ".irp idx,0,1,2,3\n"\
    ".ifeqs \"\\opm\",\"v\\reg\\.4b[\\idx\\]\"\n"\
    ".set vm,\\reg\n"\
    ".set h,\\idx / 2\n"\
    ".set l,\\idx %% 2\n"\
    ".endif\n"\
    ".endr\n"\
    ".endr\n"\
    ".ifndef vd\n"\
    ".error \"Bad operand \\opd\"\n"\
    ".exitm\n"\
    ".endif\n"\
    ".ifndef vn\n"\
    ".error \"Bad operand \\opn\"\n"\
    ".exitm\n"\
    ".endif\n"\
    ".ifndef vm\n"\
    ".error \"Bad operand \\opm\"\n"\
    ".exitm\n"\
    ".endif\n"\
    ".ifndef h\n"\
    ".error \"Bad operand \\opm\"\n"\
    ".exitm\n"\
    ".endif\n"\
    ".ifndef l\n"\
    ".error \"Bad operand \\opm\"\n"\
    ".exitm\n"\
    ".endif\n"\
    ".int	 0x6f80e000 | vd | (vn << 5) | (vm << 16) | (l << 21) | (h << 11)\n"\
    ".endm\n"\

