/*
 * Copyright (c) 2018 ARM Limited.
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

#pragma once

#include <fcntl.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <thread>

extern int l1_cache_size;
extern int l2_cache_size;
extern int force_cpu;

#ifdef __ANDROID__
inline unsigned long      stoul( const std::string& str, std::size_t* pos = 0, int base = 10 )
{
        char *end;
        const unsigned long ret = strtoul( str.c_str(), &end, base);
        *pos = end - str.c_str();
        return ret;
}
inline int       stoi( const std::string& str, std::size_t* pos = 0, int base = 10 )
{
        return atoi(str.c_str());        
}
#endif


#if ! defined(BARE_METAL) && (defined(__arm__) || defined(__aarch64__))
#include <sys/auxv.h>

/* Get HWCAP bits from asm/hwcap.h */
#include <asm/hwcap.h>
#endif /* !BARE_METAL */

/* Make sure the bits we care about are defined, just in case asm/hwcap.h is
 * out of date (or for bare metal mode) */
#ifndef HWCAP_ASIMDHP
#define HWCAP_ASIMDHP      (1 << 10)
#endif

#ifndef HWCAP_CPUID
#define HWCAP_CPUID        (1 << 11)
#endif

#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP      (1 << 20)
#endif

#define CPUINFO_HACK

//unsigned int get_cpu_impl();


/* CPU models - we only need to detect CPUs we have
 * microarchitecture-specific code for.
 *
 * Architecture features are detected via HWCAPs.
 */
enum class CPUModel {
    GENERIC    = 0x0001,
    A53        = 0x0010,
    A55r0      = 0x0011,
    A55r1      = 0x0012,
};

class CPUInfo
{
private:
    struct PerCPUData {
        CPUModel  model      = CPUModel::GENERIC;
        uint32_t  midr       = 0;
        bool      model_set  = false;
    };

    std::vector<PerCPUData> _percpu={};

    bool _cpuid   = false;
    bool _fp16    = false;
    bool _dotprod = false;

    unsigned int L1_cache_size = 32768;
    unsigned int L2_cache_size = 262144;

    /* Convert an MIDR register value to a CPUModel enum value. */
    CPUModel midr_to_model(const unsigned int midr) const {
        CPUModel model;

        // Unpack variant and CPU ID
        int variant = (midr >> 20) & 0xF;
        int cpunum = (midr >> 4) & 0xFFF;

        /* Only CPUs we have code paths for are detected.  All other CPUs
         * can be safely classed as "GENERIC"
         */

        switch(cpunum) {
            case 0xd03:
                model = CPUModel::A53;
                break;

            case 0xd05:
                if (variant) {
                    model = CPUModel::A55r1;
                } else {
                    model = CPUModel::A55r0;
                }
                break;

            default:
                model = CPUModel::GENERIC;
                break;
        }

        return model;
    }

    /* If the CPUID capability is present, MIDR information is provided in
       /sys.  Use that to populate the CPU model table.  */
    void populate_models_cpuid() {
        for (unsigned long int i=0; i<_percpu.size(); i++) {
            std::stringstream str;
            str << "/sys/devices/system/cpu/cpu" << i << "/regs/identification/midr_el1";
            std::ifstream file;

            file.open(str.str(), std::ios::in);

            if (file.is_open()) {
                std::string line;

                if (bool(getline(file, line))) {
                    const unsigned long midr = stoul(line, nullptr, 16);

                    _percpu[i].midr      = (midr & 0xffffffff);
                    _percpu[i].model     = midr_to_model(_percpu[i].midr);
                    _percpu[i].model_set = true;
                }
            }
        }
    }

    /* If "long-form" cpuinfo is present, parse that to populate models. */
    void populate_models_cpuinfo() {
        std::regex   proc_regex("^processor.*(\\d+)$");
        std::regex   imp_regex("^CPU implementer.*0x(..)$");
        std::regex   var_regex("^CPU variant.*0x(.)$");
        std::regex   part_regex("^CPU part.*0x(...)$");
        std::regex   rev_regex("^CPU revision.*(\\d+)$");

        std::ifstream file;
        file.open("/proc/cpuinfo", std::ios::in);

        if (file.is_open()) {
            std::string line;
            int midr=0;
            int curcpu=-1;

            while(bool(getline(file, line))) {
                std::smatch match;

                if (std::regex_match(line, match, proc_regex)) {
                    std::string id = match[1];
                    int newcpu=stoi(id, nullptr, 0);

                    if (curcpu >= 0 && midr==0) {
                        // Matched a new CPU ID without any description of the previous one - looks like old format.
                        return;
                    }

                    if (curcpu >= 0) {
                        _percpu[curcpu].midr      = midr;
                        _percpu[curcpu].model     = midr_to_model(midr);
                        _percpu[curcpu].model_set = true;
                    }

                    midr=0;
                    curcpu=newcpu;

                    continue;
                }

                if (std::regex_match(line, match, imp_regex)) {
                    int impv = stoi(match[1], nullptr, 16);
                    midr |= (impv << 24);
                    continue;
                }

                if (std::regex_match(line, match, var_regex)) {
                    int varv = stoi(match[1], nullptr, 16);
                    midr |= (varv << 16);
                    continue;
                }

                if (std::regex_match(line, match, part_regex)) {
                    int partv = stoi(match[1], nullptr, 16);
                    midr |= (partv << 4);
                    continue;
                }

                if (std::regex_match(line, match, rev_regex)) {
                    int regv = stoi(match[1], nullptr, 10);
                    midr |= (regv);
                    midr |= (0xf << 16);
                    continue;
                }
            }

            if (curcpu >= 0) {
                _percpu[curcpu].midr      = midr;
                _percpu[curcpu].model     = midr_to_model(midr);
                _percpu[curcpu].model_set = true;

            }
        }
    }

    /* Identify the maximum valid CPUID in the system.  This reads
     * /sys/devices/system/cpu/present to get the information.  */
    int get_max_cpus() {
        int max_cpus = 1;

#if ! defined(BARE_METAL) && (defined(__arm__) || defined(__aarch64__))
        std::ifstream CPUspresent;
        CPUspresent.open("/sys/devices/system/cpu/present", std::ios::in);
        bool success = false;

        if (CPUspresent.is_open()) {
            std::string line;

            if (bool(getline(CPUspresent, line))) {
                /* The content of this file is a list of ranges or single values, e.g.
                 * 0-5, or 1-3,5,7 or similar.  As we are interested in the
                 * max valid ID, we just need to find the last valid
                 * delimiter ('-' or ',') and parse the integer immediately after that.
                 */
                auto startfrom=line.begin();

                for (auto i=line.begin(); i<line.end(); ++i) {
                    if (*i=='-' || *i==',') {
                        startfrom=i+1;
                    }
                }

                line.erase(line.begin(), startfrom);

                max_cpus = stoi(line, nullptr, 0) + 1;
                success = true;
            }
        }

        // Return std::thread::hardware_concurrency() as a fallback.
        if (!success) {
            max_cpus = std::thread::hardware_concurrency();
        }
#endif // !BARE_METAL

        return max_cpus;
    }

public:
    CPUInfo() {
#if ! defined(BARE_METAL) && (defined(__arm__) || defined(__aarch64__))
        unsigned long hwcaps = getauxval(AT_HWCAP);

        if (hwcaps & HWCAP_CPUID) {
            _cpuid = true;
        }

        if (hwcaps & HWCAP_ASIMDHP) {
            _fp16 = true;
        }

        if (hwcaps & HWCAP_ASIMDDP) {
            _dotprod = true;
        }

#ifdef __aarch64__
        /* Pre-4.15 kernels don't have the ASIMDDP bit.
         *
         * Although the CPUID bit allows us to read the feature register
         * directly, the kernel quite sensibly masks this to only show
         * features known by it to be safe to show to userspace.  As a
         * result, pre-4.15 kernels won't show the relevant bit in the
         * feature registers either.
         *
         * So for now, use a whitelist of CPUs known to support the feature.
         */
        if (!_dotprod && _cpuid) {
            /* List of CPUs with dot product support:         A55r1       A75r1       A75r2  */
            const unsigned int dotprod_whitelist_masks[]  = { 0xfff0fff0, 0xfff0fff0, 0xfff0fff0, 0 };
            const unsigned int dotprod_whitelist_values[] = { 0x4110d050, 0x4110d0a0, 0x4120d0a0, 0 };

            unsigned long cpuid;

            __asm __volatile (
                "mrs %0, midr_el1\n"
                : "=r" (cpuid)
                :
                :
            );

            for (int i=0;dotprod_whitelist_values[i];i++) {
                if ((cpuid & dotprod_whitelist_masks[i]) == dotprod_whitelist_values[i]) {
                    _dotprod = true;
                    break;
                }
            }
        }
#endif
        _percpu.resize(get_max_cpus());
#endif
        if (_cpuid) {
            populate_models_cpuid();
        } else {
            populate_models_cpuinfo();
        }
    }

    void set_fp16(const bool fp16) {
        _fp16 = fp16;
    }

    void set_dotprod(const bool dotprod) {
        _dotprod = dotprod;
    }

    void set_cpu_model(unsigned long cpuid, CPUModel model) {
        if (_percpu.size() > cpuid) {
            _percpu[cpuid].model     = model;
            _percpu[cpuid].model_set = true;
        }
    }

    bool has_fp16() const {
        return _fp16;
    }

    bool has_dotprod() const {
        return _dotprod;
    }

    CPUModel get_cpu_model(unsigned long cpuid) const {
        if (cpuid < _percpu.size()) {
            return _percpu[cpuid].model;
        }

        return CPUModel::GENERIC;
    }

    CPUModel get_cpu_model() const {
#if defined(BARE_METAL) || (!defined(__arm__) && !defined( __aarch64__) )
        return get_cpu_model(0);
#else
        return get_cpu_model(sched_getcpu());
#endif
    }

    unsigned int get_L1_cache_size() const {
        return L1_cache_size;
    }

    void set_L1_cache_size(unsigned int size) {
        L1_cache_size = size;
    }

    unsigned int get_L2_cache_size() const {
        return L2_cache_size;
    }

    void set_L2_cache_size(unsigned int size) {
        L2_cache_size = size;
    }
};

CPUInfo *get_CPUInfo();
