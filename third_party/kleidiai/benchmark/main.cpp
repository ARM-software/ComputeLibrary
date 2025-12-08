//
// SPDX-FileCopyrightText: Copyright 2024-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>

#include "benchmark/imatmul/imatmul_registry.hpp"
#include "benchmark/matmul/matmul_registry.hpp"
#include "kai/kai_common.h"

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wswitch-default"
#endif  // __GNUC__

#include <benchmark/benchmark.h>

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif  // __GNUC__

namespace {

void print_matmul_usage(const char* name, bool defaulted = false) {
    std::ostringstream oss;
    if (defaulted) {
        oss << "Warning: No operation specified, defaulting to 'matmul' mode.\n";
        oss << "If you intended to run a different operation, specify it explicitly like so:\n";
        oss << '\t' << name << " imatmul [options]\n\n";
    }
    oss << "Matmul usage:" << '\n';
    oss << '\t' << name << " matmul -m <M> -n <N> -k <K> [-b <block_size>]" << '\n';
    oss << "Options:" << '\n';
    oss << "\t-m,-n,-k\tMatrix dimensions (LHS MxK, RHS KxN)" << '\n';
    oss << "\t-b\t\t(Optional) Block size for blockwise quantization" << '\n';
    std::cerr << oss.str() << '\n';
}

void print_imatmul_usage(const char* name) {
    std::ostringstream oss;
    oss << "IndirectMatmul usage:" << '\n';
    oss << '\t' << name << " imatmul -m <M> -n <N> -c <k_chunk_count> -l <k_chunk_length>" << '\n';
    oss << "Options:" << '\n';
    oss << "\t-m\tNumber of rows (LHS)" << '\n';
    oss << "\t-n\tNumber of columns (RHS)" << '\n';
    oss << "\t-c\tK chunk count" << '\n';
    oss << "\t-l\tK chunk length" << '\n';
    std::cerr << oss.str() << '\n';
}

void print_global_usage(const char* name) {
    std::ostringstream oss;
    oss << "Usage:" << '\n';
    oss << '\t' << name << " <matmul|imatmul> [options]" << '\n';
    oss << "\nIf no operation is provided, defaults to: " << name << " matmul [options]" << '\n';
    oss << "\nBenchmark Framework options:" << '\n';
    oss << '\t' << name << " --help" << '\n';
    std::cerr << oss.str() << '\n';

    print_matmul_usage(name);
    print_imatmul_usage(name);
}

}  // namespace

static std::optional<std::string> find_user_benchmark_filter(int argc, char** argv) {
    static constexpr std::string_view benchmark_filter_eq = "--benchmark_filter=";
    static constexpr std::string_view benchmark_filter = "--benchmark_filter";

    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];
        if (!arg) {
            continue;
        }

        // --benchmark_filter=REGEX
        std::string_view arg_view(arg);
        if (arg_view.substr(0, benchmark_filter_eq.length()) == benchmark_filter_eq) {
            auto val = arg_view.substr(benchmark_filter_eq.length());
            return std::string(val);
        }

        // --benchmark_filter REGEX
        if (arg_view == benchmark_filter && i + 1 < argc) {
            const char* val = argv[i + 1];
            return std::string(val ? val : "");
        }
    }
    return std::nullopt;
}

static int run_matmul(
    int argc, char** argv, bool default_to_matmul, const std::optional<std::string>& user_filter_opt) {
    bool mflag = false, nflag = false, kflag = false, bflag = false;
    size_t m = 1, n = 1, k = 1, bl = 32;

    optind = 1;
    int opt;
    while ((opt = getopt(argc, argv, "m:n:k:b:")) != -1) {
        switch (opt) {
            case 'm':
                m = std::atoi(optarg);
                mflag = true;
                break;
            case 'n':
                n = std::atoi(optarg);
                nflag = true;
                break;
            case 'k':
                k = std::atoi(optarg);
                kflag = true;
                break;
            case 'b':
                bl = std::atoi(optarg);
                bflag = true;
                break;
            default:
                print_matmul_usage(argv[0], default_to_matmul);
                return EXIT_FAILURE;
        }
    }

    if (!mflag || !nflag || !kflag) {
        print_matmul_usage(argv[0]);
        return EXIT_FAILURE;
    }
    if (!bflag) {
        std::cerr << "Optional argument -b not specified. Defaulting to block size " << bl << "\n";
    }

    kai::benchmark::RegisterMatMulBenchmarks({m, n, k}, bl);

    // Default filter if user didn’t supply one
    std::string spec = user_filter_opt.has_value() ? *user_filter_opt : std::string("^kai_matmul");

    ::benchmark::RunSpecifiedBenchmarks(nullptr, nullptr, spec);
    ::benchmark::Shutdown();
    return 0;
}

static int run_imatmul(int argc, char** argv, const std::optional<std::string>& user_filter_opt) {
    bool mflag = false, nflag = false, cflag = false, lflag = false;
    size_t m = 1, n = 1, k_chunk_count = 1, k_chunk_length = 1;

    optind = 1;
    int opt;
    while ((opt = getopt(argc, argv, "m:n:c:l:")) != -1) {
        switch (opt) {
            case 'm':
                m = std::atoi(optarg);
                mflag = true;
                break;
            case 'n':
                n = std::atoi(optarg);
                nflag = true;
                break;
            case 'c':
                k_chunk_count = std::atoi(optarg);
                cflag = true;
                break;
            case 'l':
                k_chunk_length = std::atoi(optarg);
                lflag = true;
                break;
            default:
                print_imatmul_usage(argv[0]);
                return EXIT_FAILURE;
        }
    }

    if (!mflag || !nflag || !cflag || !lflag) {
        print_imatmul_usage(argv[0]);
        return EXIT_FAILURE;
    }

    std::cerr << "Running imatmul benchmarks with m=" << m << ", n=" << n << ", k_chunk_count=" << k_chunk_count
              << ", k_chunk_length=" << k_chunk_length << "\n";

    kai::benchmark::RegisteriMatMulBenchmarks(m, n, k_chunk_count, k_chunk_length);

    // Default filter if user didn’t supply one
    std::string spec = user_filter_opt.has_value() ? *user_filter_opt : std::string("^kai_imatmul");

    ::benchmark::RunSpecifiedBenchmarks(nullptr, nullptr, spec);
    ::benchmark::Shutdown();
    return 0;
}

int main(int argc, char** argv) {
    // Detect user-provided filter BEFORE Initialize() consumes the benchmark framework flags
    const auto user_filter_opt = find_user_benchmark_filter(argc, argv);

    // Check for --benchmark_list_tests in argv
    bool list_tests = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strstr(argv[i], "--benchmark_list_tests") == argv[i]) {
            list_tests = true;
            break;
        }
    }

    ::benchmark::Initialize(&argc, argv);

    std::cerr << "KleidiAI version: v" << kai_get_version() << "\n";

    // Determine subcommand (mode): matmul or imatmul.
    enum class Mode : uint8_t { COMPAT, MATMUL, IMATMUL } mode = Mode::COMPAT;

    static constexpr std::string_view MATMUL = "matmul";
    static constexpr std::string_view IMATMUL = "imatmul";

    std::vector<std::string_view> args(argv, argv + argc);

    if (!list_tests && argc < 2) {
        print_global_usage(argv[0]);
        return EXIT_FAILURE;
    }

    if (argc >= 2 && args[1] == MATMUL) {
        mode = Mode::MATMUL;
        argv += 1;
        argc -= 1;
    } else if (argc >= 2 && args[1] == IMATMUL) {
        mode = Mode::IMATMUL;
        argv += 1;
        argc -= 1;
    }

    if (list_tests) {
        std::string spec;
        if (mode == Mode::COMPAT) {
            kai::benchmark::RegisterMatMulBenchmarks({1, 1, 1}, 32);
            kai::benchmark::RegisteriMatMulBenchmarks(1, 1, 1, 1);
            spec = user_filter_opt.value_or("");
        } else if (mode == Mode::MATMUL) {
            kai::benchmark::RegisterMatMulBenchmarks({1, 1, 1}, 32);
            spec = user_filter_opt.has_value() ? *user_filter_opt : std::string("^kai_matmul");
        } else if (mode == Mode::IMATMUL) {
            kai::benchmark::RegisteriMatMulBenchmarks(1, 1, 1, 1);
            spec = user_filter_opt.has_value() ? *user_filter_opt : std::string("^kai_imatmul");
        }
        ::benchmark::RunSpecifiedBenchmarks(nullptr, nullptr, spec);
        ::benchmark::Shutdown();
        return 0;
    }

    switch (mode) {
        case Mode::COMPAT:
            return run_matmul(argc, argv, true, user_filter_opt);
        case Mode::MATMUL:
            return run_matmul(argc, argv, false, user_filter_opt);
        case Mode::IMATMUL:
            return run_imatmul(argc, argv, user_filter_opt);
        default:
            print_global_usage(argv[0]);
            return EXIT_FAILURE;
    }
}
