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
#pragma once

#ifdef CYCLE_PROFILING

#include "../perf.h"

class profiler {
private:
    static const int maxevents = 10000;
    unsigned long times[maxevents];
    unsigned long units[maxevents];
    int events[maxevents];
    int currentevent;
    int countfd;

public:
    profiler() {
        currentevent=0;
        countfd=open_cycle_counter();
    }

    ~profiler() {
        close(countfd);
        int tots[5];
        unsigned long counts[5];
        unsigned long tunits[5];
        const char * descs[] = { "Prepare A", "Prepare B", "Kernel", "Merge" };

        for (int i=1; i<5; i++) {
            tots[i] = 0;
            counts[i] = 0;
            tunits[i] = 0;
        }

        printf("Profiled events:\n");
        for (int i=0; i<currentevent; i++) {
            tots[events[i]]++;
            counts[events[i]] += times[i];
            tunits[events[i]] += units[i];
        }

        printf("%20s  %9s %9s %9s %12s %9s\n", "", "Events", "Total", "Average", "Bytes/MACs", "Per cycle");
        for (int i=1; i<5; i++) {
            printf("%20s: %9d %9ld %9ld %12lu %9.2f\n",descs[i-1],tots[i],counts[i],counts[i]/tots[i],tunits[i],(float)tunits[i]/counts[i]);
        }
    }

    template <typename T>
    void operator() (int i, unsigned long u, T func) {
        if (currentevent==maxevents) {
            func();
        } else {
            events[currentevent] = i;
            units[currentevent] = u;
            start_counter(countfd);
            func();
            long long cycs = stop_counter(countfd);
            times[currentevent++] = cycs;
        }
    }
};

#else

class profiler {
public:
    template <typename T>
    void operator() (int i, unsigned long u, T func) {
        func();
    }
};

#endif

#define PROFILE_PREPA 1
#define PROFILE_PREPB 2
#define PROFILE_KERNEL 3
#define PROFILE_MERGE 4


