#!/usr/bin/env python
#FIXME: Remove this file before the release

import os
import re
import sys

def get_list_includes():
    return "include . 3rdparty/include kernels computer_vision".split()

def get_list_flags( filename, arch):
    assert arch in ["armv7", "aarch64"]
    flags = ["-std=c++11"]
    if "tests/validation_old" in filename:
        flags.append("-DBOOST")
    flags.append("-DARM_COMPUTE_CPP_SCHEDULER=1")
    if arch == "aarch64":
        flags.append("-DARM_COMPUTE_CL")
        flags.append("-DARM_COMPUTE_ENABLE_FP16")
    return flags

def filter_files( list_files ):
    to_check = []
    for f in list_files:
        if os.path.splitext(f)[1] != ".cpp":
            continue
        if "computer_vision" in f:
            continue
        if "openvx-arm_compute" in f:
            continue
        if "tests/validation_old" in f:
            continue
        # Skip OMPScheduler as it causes problems in clang
        if "OMPScheduler.cpp" in f:
            continue
        to_check.append(f)
    return to_check

def filter_clang_tidy_lines( lines ):
    out = []
    print_context=False
    for i in range(0, len(lines)):
        line = lines[i]

        if "error:" in line:
            if (("Utils.cpp" in line and "'arm_compute_version.embed' file not found" in line) or
                ("cl2.hpp" in line and "cast from pointer to smaller type 'cl_context_properties' (aka 'int') loses information" in line) or
                ("arm_fp16.h" in line) or
                ("omp.h" in line) or
                ("memory" in line and "cast from pointer to smaller type 'uintptr_t' (aka 'unsigned int') loses information" in line) or
                ("NEMath.inl" in line and "statement expression not allowed at file scope" in line) or
                ("Utils.h" in line and "no member named 'unmap' in 'arm_compute::Tensor'" in line) or
                ("Utils.h" in line and "no member named 'map' in 'arm_compute::Tensor'" in line) or
                "3rdparty" in line):
                print_context=False
                continue

            out.append(line)
            print_context=True
        elif "warning:" in line:
            if ("uninitialized record type: '__ret'" in line or
               "local variable '__bound_functor' is still referred to by the global variable '__once_callable'" in line or
               ("Error.cpp" in line and "thrown exception type is not nothrow copy constructible" in line) or
               ("Error.cpp" in line and "uninitialized record type: 'args'" in line) or
               ("Error.cpp" in line and "do not call c-style vararg functions" in line) or
               ("Error.cpp" in line and "do not define a C-style variadic function" in line) or
               ("NEMinMaxLocationKernel.cpp" in line and "move constructors should be marked noexcept" in line) or
               ("NEMinMaxLocationKernel.cpp" in line and "move assignment operators should be marked noexcept" in line) or
               ("PMUCounter.cpp" in line and "consider replacing 'long long' with 'int64'" in line) or
               ("Validation.cpp" in line and "parameter 'classified_labels' is unused" in line) or
               ("Validation.cpp" in line and "parameter 'expected_labels' is unused" in line) or
               ("Reference.cpp" in line and "parameter 'rois' is unused" in line) or
               ("Reference.cpp" in line and "parameter 'shapes' is unused" in line) or
               ("Reference.cpp" in line and re.search(r"parameter '[^']+' is unused", line)) or
               ("ReferenceCPP.cpp" in line and "parameter 'rois' is unused" in line) or
               ("ReferenceCPP.cpp" in line and "parameter 'srcs' is unused" in line) or
               ("ReferenceCPP.cpp" in line and re.search(r"parameter '[^']+' is unused", line)) or
               ("NEGEMMMatrixMultiplyKernel.cpp" in line and "do not use C-style cast to convert between unrelated types" in line) or
               ("NEPoolingLayerKernel.cpp" in line and "do not use C-style cast to convert between unrelated types" in line) or
               ("NESoftmaxLayerKernel.cpp" in line and "do not use C-style cast to convert between unrelated types" in line) or
               ("GraphUtils.cpp" in line and "consider replacing 'unsigned long' with 'uint64'" in line) or
               ("parameter 'memory_manager' is unused" in line) or
               ("parameter 'memory_manager' is copied for each invocation but only used as a const reference" in line) or
               "3rdparty" in line):
                print_context=False
                continue

            if "do not use C-style cast to convert between unrelated types" in line:
                if i + 1 < len(lines) and "vgetq_lane_f16" in lines[i + 1]:
                    print_context=False
                    continue

            if "use 'using' instead of 'typedef'" in line:
                if i + 1 < len(lines) and "BOOST_FIXTURE_TEST_SUITE" in lines[i + 1]:
                    print_context=False
                    continue

            if "do not call c-style vararg functions" in line:
                if (i + 1 < len(lines) and
                   ("BOOST_TEST" in lines[i + 1] or
                    "BOOST_FAIL" in lines[i + 1] or
                    "BOOST_CHECK_THROW" in lines[i + 1] or
                    "syscall" in lines[i + 1])):
                        print_context=False
                        continue

            out.append(line)
            print_context=True
        elif print_context:
            out.append(line)

    return out

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: {} CLANG-TIDY_OUTPUT_FILE".format(sys.argv[0]))
        sys.exit(1)

    errors = []
    with open(sys.argv[1], mode="r") as clang_tidy_file:
        lines = clang_tidy_file.readlines()
        errors = filter_clang_tidy_lines(lines)
        print "\n".join(errors)

    sys.exit(0 if len(errors) == 0 else 1)
