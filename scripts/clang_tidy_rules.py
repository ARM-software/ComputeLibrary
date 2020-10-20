#!/usr/bin/env python3
import os
import re
import sys

def get_list_includes():
    return "arm_compute/core/NEON/kernels/assembly " \
           "arm_compute/core/NEON/kernels/convolution/common " \
           "arm_compute/core/NEON/kernels/convolution/depthwise " \
           "arm_compute/core/NEON/kernels/convolution/winograd " \
           "src/core/NEON/kernels/assembly " \
           "src/core/NEON/kernels/convolution/winograd " \
           "include/linux include " \
           ". " \
           "3rdparty/include kernels".split()

def get_list_flags( filename, arch):
    assert arch in ["armv7", "aarch64"]
    flags = ["-std=c++11"]
    flags.append("-DARM_COMPUTE_CPP_SCHEDULER=1")
    flags.append("-DARM_COMPUTE_CL")
    flags.append("-DARM_COMPUTE_GC")
    if arch == "aarch64":
        flags.append("-DARM_COMPUTE_AARCH64_V8_2")
    return flags

def filter_files( list_files ):
    to_check = []
    for f in list_files:
        if os.path.splitext(f)[1] != ".cpp":
            continue
        # Skip OMPScheduler as it causes problems in clang
        if (("OMPScheduler.cpp" in f) or
            ("CLTracePoint.cpp" in f) or
            ("NETracePoint.cpp" in f) or
            ("TracePoint.cpp" in f)):
            continue
        to_check.append(f)
    return to_check

def filter_clang_tidy_lines( lines ):
    out = []
    print_context=False
    for i in range(0, len(lines)):
        line = lines[i]

        if "/arm_gemm/" in line:
            continue

        if "/convolution/" in line:
            continue

        if "/validate_examples/" in line:
            continue

        if "error:" in line:
            if (("Version.cpp" in line and "arm_compute_version.embed" in line and "file not found" in line) or
                ("arm_fp16.h" in line) or
                ("omp.h" in line) or
                ("cast from pointer to smaller type 'uintptr_t' (aka 'unsigned int') loses information" in line) or
                ("cast from pointer to smaller type 'cl_context_properties' (aka 'int') loses information" in line) or
                ("cast from pointer to smaller type 'std::uintptr_t' (aka 'unsigned int') loses information" in line) or
                ("NEMath.inl" in line and "statement expression not allowed at file scope" in line) or
                ("Utils.h" in line and "no member named 'unmap' in 'arm_compute::Tensor'" in line) or
                ("Utils.h" in line and "no member named 'map' in 'arm_compute::Tensor'" in line) or
                ("CPUUtils.cpp" in line and "'asm/hwcap.h' file not found" in line) or
                "3rdparty" in line or
                ("'arm_compute_version.embed' file not found" in line) ):
                print_context=False
                continue

            out.append(line)
            print_context=True
        elif "warning:" in line:
            if ("uninitialized record type: '__ret'" in line or
               "local variable '__bound_functor' is still referred to by the global variable '__once_callable'" in line or
               "assigning newly created 'gsl::owner<>'" in line or
               "calling legacy resource function without passing a 'gsl::owner<>'" in line or
               "deleting a pointer through a type that is not marked 'gsl::owner<>'" in line or
               (any(f in line for f in ["Error.cpp","Error.h"]) and "thrown exception type is not nothrow copy constructible" in line) or
               (any(f in line for f in ["Error.cpp","Error.h"]) and "uninitialized record type: 'args'" in line) or
               (any(f in line for f in ["Error.cpp","Error.h"]) and "do not call c-style vararg functions" in line) or
               (any(f in line for f in ["Error.cpp","Error.h"]) and "do not define a C-style variadic function" in line) or
               ("TensorAllocator.cpp" in line and "warning: pointer parameter 'ptr' can be pointer to const" in line) or
               ("TensorAllocator.cpp" in line and "warning: do not declare C-style arrays" in line) or
               ("RawTensor.cpp" in line and "warning: pointer parameter 'ptr' can be pointer to const" in line) or
               ("RawTensor.cpp" in line and "warning: do not declare C-style arrays" in line) or
               ("GCBufferAllocator.cpp" in line and "warning: initializing non-owner" in line) or
               ("NEMinMaxLocationKernel.cpp" in line and "move constructors should be marked noexcept" in line) or
               ("NEMinMaxLocationKernel.cpp" in line and "move assignment operators should be marked noexcept" in line) or
               ("CLMinMaxLocationKernel.cpp" in line and "Forming reference to null pointer" in line) or
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
               ("NESoftmaxLayerKernel.cpp" in line and "macro argument should be enclosed in parentheses" in line) or
               ("GraphUtils.cpp" in line and "consider replacing 'unsigned long' with 'uint32'" in line) or
               ("GraphUtils.cpp" in line and "consider replacing 'unsigned long' with 'uint64'" in line) or
               ("ConvolutionLayer.cpp" in line and "move assignment operators should be marked noexcept" in line) or
               ("ConvolutionLayer.cpp" in line and "move constructors should be marked noexcept" in line) or
               ("parameter 'memory_manager' is unused" in line) or
               ("parameter 'memory_manager' is copied for each invocation but only used as a const reference" in line) or
               ("DeconvolutionLayer.cpp" in line and "casting (double + 0.5) to integer leads to incorrect rounding; consider using lround" in line) or
               ("NEWinogradLayerKernel.cpp" in line and "use '= default' to define a trivial destructor" in line) or
               ("NEGEMMLowpMatrixMultiplyCore.cpp" in line and "constructor does not initialize these fields" in line) or
               ("NEGEMMLowpAssemblyMatrixMultiplyCore" in line and "constructor does not initialize these fields" in line) or
               ("NEDepthwiseConvolutionLayerNativeKernel" in line and re.search(r"parameter '[^']+' is unused", line)) or
               ("NEDepthwiseConvolutionAssemblyDispatch" in line and re.search(r"parameter '[^']+' is unused", line)) or
               ("CPUUtils.cpp" in line and "consider replacing 'unsigned long' with 'uint64'" in line) or
               ("CPUUtils.cpp" in line and "parameter 'cpusv' is unused" in line) or
               ("CPUUtils.cpp" in line and "warning: uninitialized record type" in line) or
               ("GCKernelLibrary.cpp" in line and "warning: do not declare C-style arrays" in line) or
               ("Utils.h" in line and "warning: Use of zero-allocated memory" in line) or
               ("NEDepthwiseConvolutionLayerNativeKernel.cpp" in line and "misc-non-private-member-variables-in-classes" in line) or # This is to prevent false positive, should be reassessed with the newer clang-tidy
               ("NEDepthwiseConvolutionLayerNativeKernel.cpp" in line and "cppcoreguidelines-pro-type-member-init" in line) or # This is to prevent false positive, should be reassessed with the newer clang-tidy
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
                    "ARM_COMPUTE_ERROR_VAR" in lines[i + 1] or
                    "ARM_COMPUTE_RETURN_ON" in lines[i + 1] or
                    "syscall" in lines[i + 1])):
                        print_context=False
                        continue

            out.append(line)
            print_context=True
        elif (("CLMinMaxLocationKernel.cpp" in line and "'?' condition is false" in line) or
              ("CLMinMaxLocationKernel.cpp" in line and "Assuming the condition is false" in line) or
              ("CLMinMaxLocationKernel.cpp" in line and "Assuming pointer value is null" in line) or
              ("CLMinMaxLocationKernel.cpp" in line and "Forming reference to null pointer" in line)):
               print_context=False
               continue
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
        print("\n".join(errors))

    sys.exit(0 if len(errors) == 0 else 1)
