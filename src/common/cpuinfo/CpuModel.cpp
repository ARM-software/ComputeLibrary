/*
 * Copyright (c) 2021-2023 Arm Limited.
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
#include "src/common/cpuinfo/CpuModel.h"

namespace arm_compute
{
namespace cpuinfo
{
std::string cpu_model_to_string(CpuModel model)
{
    switch (model)
    {
#define X(MODEL)          \
    case CpuModel::MODEL: \
        return #MODEL;
        ARM_COMPUTE_CPU_MODEL_LIST
#undef X
        default:
        {
            return std::string("GENERIC");
        }
    };
}

bool model_supports_fp16(CpuModel model)
{
    switch (model)
    {
        case CpuModel::GENERIC_FP16:
        case CpuModel::GENERIC_FP16_DOT:
        case CpuModel::A55r1:
        case CpuModel::A510:
        case CpuModel::X1:
        case CpuModel::V1:
        case CpuModel::A64FX:
        case CpuModel::N1:
            return true;
        default:
            return false;
    }
}

bool model_supports_dot(CpuModel model)
{
    switch (model)
    {
        case CpuModel::GENERIC_FP16_DOT:
        case CpuModel::A55r1:
        case CpuModel::A510:
        case CpuModel::X1:
        case CpuModel::V1:
        case CpuModel::N1:
            return true;
        default:
            return false;
    }
}

CpuModel midr_to_model(uint32_t midr)
{
    CpuModel model = CpuModel::GENERIC;

    // Unpack variant and CPU ID
    const int implementer = (midr >> 24) & 0xFF;
    const int variant     = (midr >> 20) & 0xF;
    const int cpunum      = (midr >> 4) & 0xFFF;

    // Only CPUs we have code paths for are detected.  All other CPUs can be safely classed as "GENERIC"
    if (implementer == 0x41) // Arm CPUs
    {
        switch (cpunum)
        {
            case 0xd03: // A53
            case 0xd04: // A35
                model = CpuModel::A53;
                break;
            case 0xd05: // A55
                if (variant != 0)
                {
                    model = CpuModel::A55r1;
                }
                else
                {
                    model = CpuModel::A55r0;
                }
                break;
            case 0xd09: // A73
                model = CpuModel::A73;
                break;
            case 0xd0a: // A75
                if (variant != 0)
                {
                    model = CpuModel::GENERIC_FP16_DOT;
                }
                else
                {
                    model = CpuModel::GENERIC_FP16;
                }
                break;
            case 0xd0c: // N1
                model = CpuModel::N1;
                break;
            case 0xd06: // A65
            case 0xd0b: // A76
            case 0xd0d: // A77
            case 0xd0e: // A76AE
            case 0xd41: // A78
            case 0xd42: // A78AE
            case 0xd4a: // E1
                model = CpuModel::GENERIC_FP16_DOT;
                break;
            case 0xd40: // V1
                model = CpuModel::V1;
                break;
            case 0xd44: // X1
                model = CpuModel::X1;
                break;
            case 0xd46: // A510
            case 0xd80: // A520
                model = CpuModel::A510;
                break;
            case 0xd15: // R82
                model = CpuModel::A55r1;
                break;
            default:
                model = CpuModel::GENERIC;
                break;
        }
    }
    else if (implementer == 0x46)
    {
        switch (cpunum)
        {
            case 0x001: // A64FX
                model = CpuModel::A64FX;
                break;
            default:
                model = CpuModel::GENERIC;
                break;
        }
    }
    else if (implementer == 0x48)
    {
        switch (cpunum)
        {
            case 0xd40: // A76
                model = CpuModel::GENERIC_FP16_DOT;
                break;
            default:
                model = CpuModel::GENERIC;
                break;
        }
    }
    else if (implementer == 0x51)
    {
        switch (cpunum)
        {
            case 0x800: // A73
                model = CpuModel::A73;
                break;
            case 0x801: // A53
                model = CpuModel::A53;
                break;
            case 0x803: // A55r0
                model = CpuModel::A55r0;
                break;
            case 0x804: // A76
                model = CpuModel::GENERIC_FP16_DOT;
                break;
            case 0x805: // A55r1
                model = CpuModel::A55r1;
                break;
            default:
                model = CpuModel::GENERIC;
                break;
        }
    }

    return model;
}
} // namespace cpuinfo
} // namespace arm_compute
