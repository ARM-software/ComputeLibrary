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

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

#include "perf.h"
#include <unistd.h>

#ifdef CYCLE_PROFILING
class EventIDContainer
{
  public:
  EventIDContainer() : container_lock(), event_ids()
  {
  }

  int get_event_id(const char *id)
  {
    std::lock_guard<std::mutex> lock(container_lock);
    if (!event_ids.count(id)) {
      event_ids.emplace(id, event_ids.size());
    }
    return event_ids[id];
  }

  unsigned int size() const
  {
    return event_ids.size();
  }

  auto begin()
  {
    return event_ids.begin();
  }

  auto end()
  {
    return event_ids.end();
  }

  private:
  std::mutex container_lock;
  std::map<const char *, int> event_ids;
};


class ThreadEventCounterContainer
{
  public:
  ThreadEventCounterContainer() : container_lock(), thread_counter_fds()
  {
  }

  int get_counter_fd()
  {
    const auto id = std::this_thread::get_id();
    std::lock_guard<std::mutex> lock(container_lock);
    if (!thread_counter_fds.count(id))
    {
      thread_counter_fds.emplace(id, open_cycle_counter());
    }
    return thread_counter_fds[id];
  }

  ~ThreadEventCounterContainer()
  {
    // Close all counter file descriptors
    for (auto& fd : thread_counter_fds)
    {
      close(fd.second);
    }
  }

  private:
  std::mutex container_lock;
  std::map<std::thread::id, int> thread_counter_fds;
};
#endif  // CYCLE_PROFILING


class profiler {
private:
#ifdef CYCLE_PROFILING
    struct ProfileEntry {
      int event_id;
      long int bytes_read, ops, bytes_written;
      long int duration;
    };

    static const int maxevents = 10000;
    ProfileEntry events[maxevents];
    int currentevent;
    std::mutex event_lock;

    EventIDContainer event_ids;
    ThreadEventCounterContainer thread_counter_fds;

    int get_event_id(const char *id)
    {
      return event_ids.get_event_id(id);
    }
#endif  // CYCLE_PROFILING

public:
#ifdef CYCLE_PROFILING
    profiler() :
      currentevent(0),
      event_lock(),
      event_ids(),
      thread_counter_fds()
    {
    }

    ~profiler() {
      std::lock_guard<std::mutex> lock_events(event_lock);

        // Compute performance from recorded events
        struct ProfileResult {
          ProfileResult() : total_calls(0),
                            total_duration(0),
                            total_bytes_read(0),
                            total_ops(0),
                            total_bytes_written(0) {
          }

          void operator+=(const ProfileEntry &rhs) {
            total_calls++;
            total_duration += rhs.duration;
            total_bytes_read += rhs.bytes_read;
            total_ops += rhs.ops;
            total_bytes_written = rhs.bytes_written;
          }

          float avg_duration(void) const {
            return static_cast<float>(total_duration) /
                   static_cast<float>(total_calls);
          }

          float bytes_read_per_cycle(void) const {
            return static_cast<float>(total_bytes_read) /
                   static_cast<float>(total_duration);
          }

          float ops_per_cycle(void) const {
            return static_cast<float>(total_ops) /
                   static_cast<float>(total_duration);
          }

          float bytes_written_per_cycle(void) const {
            return static_cast<float>(total_bytes_written) /
                   static_cast<float>(total_duration);
          }

          long int total_calls,
                   total_duration,
                   total_bytes_read,
                   total_ops,
                   total_bytes_written;
        };

        std::vector<ProfileResult> totals;
        totals.resize(event_ids.size());
        for (int i = 0; i < currentevent; i++) {
          const auto &event = events[i];
          totals[event.event_id] += event;
        }

        // Get the longest label
        int len_label = 0;
        for (const auto &kv : event_ids) {
          len_label = std::max(len_label, static_cast<int>(strlen(kv.first)));
        }

        // Get the longest values for every other field
        const auto get_length_of_field =
          [totals] (const char *title, auto f, auto len) -> size_t {
            size_t l = strlen(title);
            for (const auto &v : totals) {
              l = std::max(l, len(f(v)));
            }
            return l;
        };

        // Get the strlen for an int
        const auto intlen = [] (long int x) -> size_t {
          size_t len = 0;
          do {
            x /= 10;
            len++;
          } while (x);
          return len;
        };

        // Get the strlen for a float
        const auto floatlen = [] (const int precision) {
          return [precision] (float x) {
            size_t len = 0;

            if (!std::isfinite(x)) {
              return static_cast<size_t>(3);
            }

            do {
              x /= 10.0f;
              len++;
            } while (x > 1.0f);
            return len + 1 + precision;
          };
        };

        const int len_calls = get_length_of_field(
            "Calls", [] (const auto &v) {return v.total_calls;},
            intlen
        );
        const int len_duration = get_length_of_field(
            "Duration", [] (const auto &v) {return v.total_duration;},
            intlen
        );
        const int len_average_duration = get_length_of_field(
            "Average", [] (const auto &v) {return v.avg_duration();},
            floatlen(2)
        );
        const int len_reads_per_cycle = get_length_of_field(
            "Reads / cycle",
            [] (const auto &v) {return v.bytes_read_per_cycle();},
            floatlen(6)
        );
        const int len_ops_per_cycle = get_length_of_field(
            "Ops / cycle",
            [] (const auto &v) {return v.ops_per_cycle();},
            floatlen(6)
        );
        const int len_writes_per_cycle = get_length_of_field(
            "Writes / cycle",
            [] (const auto &v) {return v.bytes_written_per_cycle();},
            floatlen(6)
        );

        // Print header
        printf(
          "%*s    %*s    %*s    %*s    %*s    %*s    %*s\n",
          len_label, "",
          len_calls, "Calls",
          len_duration, "Duration",
          len_average_duration, "Average",
          len_reads_per_cycle, "Reads / cycle",
          len_ops_per_cycle, "Ops / cycle",
          len_writes_per_cycle, "Writes / cycle"
        );
        for (const auto &kv : event_ids) {
          const auto id = kv.second;
          printf(
            "%*s    %*ld    %*ld    %*.2f    %*.6f    %*.6f    %*.6f\n",
            len_label, kv.first,
            len_calls, totals[id].total_calls,
            len_duration, totals[id].total_duration,
            len_average_duration, totals[id].avg_duration(),
            len_reads_per_cycle, totals[id].bytes_read_per_cycle(),
            len_ops_per_cycle, totals[id].ops_per_cycle(),
            len_writes_per_cycle, totals[id].bytes_written_per_cycle()
          );
        }
        printf("\n");
    }
#endif  // CYCLE_PROFILING

    template <typename T>
    void operator() (const char * event,
                     T func,
                     long int bytes_read = 0,
                     long int ops = 0,
                     long int bytes_written = 0) {
#ifdef CYCLE_PROFILING
        if (currentevent==maxevents) {
            func();
        } else {
            const auto countfd = thread_counter_fds.get_counter_fd();
            start_counter(countfd);
            func();
            long long cycs = stop_counter(countfd);

            // Store the profiling data
            std::lock_guard<std::mutex> lock_events(event_lock);
            events[currentevent++] = {
              get_event_id(event), bytes_read, ops, bytes_written, cycs
            };
        }
#else
      (void) event;
      (void) bytes_read;
      (void) ops;
      (void) bytes_written;
      func();
#endif  // CYCLE_PROFILING
    }
};
