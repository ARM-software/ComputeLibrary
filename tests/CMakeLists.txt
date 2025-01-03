# Copyright (c) 2023-2025 Arm Limited.
#
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

file(GLOB_RECURSE validation_reference "validation/reference/*.cpp")

target_sources(
  arm_compute_validation_framework
  PRIVATE validation/Validation.cpp
          validation/Helpers.cpp
          validation/helpers/ActivationHelpers.cpp
          ${validation_reference}
          framework/Framework.cpp
          framework/Utils.cpp
          framework/Exceptions.cpp
          framework/DatasetModes.cpp
          framework/TestFilter.cpp
          framework/Profiler.cpp
          framework/ParametersLibrary.cpp
          framework/command_line/CommonOptions.cpp
          framework/instruments/WallClockTimer.cpp
          framework/instruments/InstrumentsStats.cpp
          framework/instruments/Instruments.cpp
          framework/instruments/SchedulerTimer.cpp
          framework/instruments/hwc_names.hpp
          framework/instruments/hwc.hpp
          framework/printers/PrettyPrinter.cpp
          framework/printers/Printer.cpp
          framework/printers/JSONPrinter.cpp
          framework/printers/Printers.cpp
          AssetsLibrary.cpp
          RawTensor.cpp
          main.cpp)
