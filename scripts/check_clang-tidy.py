#!/usr/bin/env python3

import sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: {} CLANG-TIDY_OUTPUT_FILE".format(sys.argv[0]))
        sys.exit(1)

    failed = False

    with open(sys.argv[1], mode="r") as clang_tidy_file:
        lines = clang_tidy_file.readlines()

        for i in range(0, len(lines)):
            line = lines[i]

            if "error:" in line:
                if (("Utils.cpp" in line and "'arm_compute_version.embed' file not found" in line) or
                    ("cl2.hpp" in line and "cast from pointer to smaller type 'cl_context_properties' (aka 'int') loses information" in line) or
                    ("memory" in line and "cast from pointer to smaller type 'uintptr_t' (aka 'unsigned int') loses information" in line) or
                    "3rdparty" in line):
                    continue

                failed = True
                print(line)
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
                   "3rdparty" in line):
                    continue

                if "do not use C-style cast to convert between unrelated types" in line:
                    if i + 1 < len(lines) and "vgetq_lane_f16" in lines[i + 1]:
                        continue

                if "use 'using' instead of 'typedef'" in line:
                    if i + 1 < len(lines) and "BOOST_FIXTURE_TEST_SUITE" in lines[i + 1]:
                        continue

                if "do not call c-style vararg functions" in line:
                    if (i + 1 < len(lines) and
                       ("BOOST_TEST" in lines[i + 1] or
                        "BOOST_FAIL" in lines[i + 1] or
                        "BOOST_CHECK_THROW" in lines[i + 1] or
                        "syscall" in lines[i + 1])):
                            continue

                failed = True
                print(line)

    sys.exit(0 if not failed else 1)
