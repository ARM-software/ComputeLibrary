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

#include <newgemm_lib.hpp>

unsigned int get_cpu_impl()
{
#ifndef BARE_METAL
    int   fd = open("/proc/cpuinfo", 0);
    char  buff[3000];
    char *pos;
    char *end;
    int   foundid = 0;
    int   variant = 0;

    int cpu = sched_getcpu();

    if(!fd)
    {
        return 0;
    }

    int charsread = read(fd, buff, 3000);
    pos           = buff;
    end           = buff + charsread;

    close(fd);

    /* So, to date I've encountered two formats for /proc/cpuinfo.
     *
     * One of them just lists processor : n  for each processor (with no
     * other info), then at the end lists part information for the current
     * CPU.
     *
     * The other has an entire clause (including part number info) for each
     * CPU in the system, with "processor : n" headers.
     *
     * We can cope with either of these formats by waiting to see
     * "processor: n" (where n = our CPU ID), and then looking for the next
     * "CPU part" field.
     */
    while(pos < end)
    {
        if(foundid && !strncmp(pos, "CPU variant", 11))
        {
            pos += 13;
            char *resume = end; // Need to continue scanning after this

            for(char *ch = pos; ch < end; ch++)
            {
                if(*ch == '\n')
                {
                    *ch    = '\0';
                    resume = ch + 1;
                    break;
                }
            }

            variant = strtoul(pos, NULL, 0);

            pos = resume;
        }

        if(foundid && !strncmp(pos, "CPU part", 8))
        {
            /* Found part number */
            pos += 11;
            unsigned int num;

            for(char *ch = pos; ch < end; ch++)
            {
                if(*ch == '\n')
                {
                    *ch = '\0';
                    break;
                }
            }

            num = strtoul(pos, NULL, 0);

            return (num << 4) | (variant << 20);
        }

        if(!strncmp(pos, "processor", 9))
        {
            /* Found processor ID, see if it's ours. */
            pos += 11;
            int num;

            for(char *ch = pos; ch < end; ch++)
            {
                if(*ch == '\n')
                {
                    *ch = '\0';
                    break;
                }
            }

            num = strtol(pos, NULL, 0);

            if(num == cpu)
            {
                foundid = 1;
            }
        }

        while(pos < end)
        {
            char ch = *pos++;
            if(ch == '\n' || ch == '\0')
            {
                break;
            }
        }
    }
#endif

    return 0;
}

CPUInfo *get_CPUInfo()
{
    static CPUInfo ci;

    return &ci;
}
