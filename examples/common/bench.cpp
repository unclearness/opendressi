#include "bench.h"

#include <spdlog/fmt/fmt.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <thread>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#elif defined(__ANDROID__)
#include <sys/system_properties.h>
#include <unistd.h>
#else
#include <sys/utsname.h>
#include <unistd.h>
#endif

namespace dressi_examples {

namespace {

// Host descriptors are best-effort ("" / 0 when unavailable) — they only
// annotate the benchmark table.

#if defined(__ANDROID__)
std::string SysProp(const char* name) {
    char buf[PROP_VALUE_MAX] = {};
    return __system_property_get(name, buf) > 0 ? std::string(buf)
                                                : std::string();
}
#endif

#if !defined(_WIN32) && !defined(__ANDROID__)
// First value of "<key><spaces>: value" in a /proc- or os-release-style file
std::string FileValue(const char* path, const std::string& key,
                      char delim = ':') {
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind(key, 0) == 0) {
            const size_t pos = line.find(delim);
            if (pos != std::string::npos) {
                std::string v = line.substr(pos + 1);
                v.erase(0, v.find_first_not_of(" \t\""));
                const size_t end = v.find_last_not_of(" \t\"");
                v.erase(end == std::string::npos ? 0 : end + 1);
                return v;
            }
        }
    }
    return {};
}
#endif

std::string OsString() {
#if defined(_WIN32)
    char product[128] = {};
    char build[32] = {};
    DWORD n = sizeof(product);
    RegGetValueA(HKEY_LOCAL_MACHINE,
                 "SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion",
                 "ProductName", RRF_RT_REG_SZ, nullptr, product, &n);
    n = sizeof(build);
    RegGetValueA(HKEY_LOCAL_MACHINE,
                 "SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion",
                 "CurrentBuildNumber", RRF_RT_REG_SZ, nullptr, build, &n);
    // ProductName still reads "Windows 10 ..." on Windows 11 (Microsoft never
    // updated the string); build >= 22000 is the real Windows 11 boundary.
    std::string name = product[0] ? product : "Windows";
    if (const int b = build[0] ? std::atoi(build) : 0; b >= 22000) {
        const size_t pos = name.find("Windows 10");
        if (pos != std::string::npos) {
            name.replace(pos, sizeof("Windows 10") - 1, "Windows 11");
        }
    }
    return fmt::format("{} (build {})", name, build[0] ? build : "?");
#elif defined(__ANDROID__)
    const std::string rel = SysProp("ro.build.version.release");
    const std::string manu = SysProp("ro.product.manufacturer");
    const std::string model = SysProp("ro.product.model");
    std::string os = "Android " + (rel.empty() ? "?" : rel);
    if (!model.empty()) {
        os += " (" + (manu.empty() ? "" : manu + " ") + model + ")";
    }
    return os;
#else
    std::string os = FileValue("/etc/os-release", "PRETTY_NAME", '=');
    if (os.empty()) {
        struct utsname un = {};
        if (uname(&un) == 0) {
            os = std::string(un.sysname) + " " + un.release;
        }
    }
    return os;
#endif
}

std::string CpuString() {
#if defined(_WIN32)
    char name[128] = {};
    DWORD n = sizeof(name);
    RegGetValueA(HKEY_LOCAL_MACHINE,
                 "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                 "ProcessorNameString", RRF_RT_REG_SZ, nullptr, name, &n);
    std::string s = name;  // the registry value is space-padded
    s.erase(s.find_last_not_of(' ') + 1);
    return s;
#elif defined(__ANDROID__)
    // The SoC identifies a phone's CPU cluster better than /proc/cpuinfo
    const std::string manu = SysProp("ro.soc.manufacturer");
    const std::string model = SysProp("ro.soc.model");
    if (!model.empty()) {
        return manu.empty() ? model : manu + " " + model;
    }
    return SysProp("ro.hardware");
#else
    return FileValue("/proc/cpuinfo", "model name");
#endif
}

double RamGb() {
#if defined(_WIN32)
    MEMORYSTATUSEX st = {};
    st.dwLength = sizeof(st);
    if (GlobalMemoryStatusEx(&st)) {
        return double(st.ullTotalPhys) / (1024.0 * 1024.0 * 1024.0);
    }
    return 0.0;
#else
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long page = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page > 0) {
        return double(pages) * double(page) / (1024.0 * 1024.0 * 1024.0);
    }
    return 0.0;
#endif
}

}  // namespace

double MedianMs(std::vector<double> samples) {
    if (samples.empty()) {
        return 0.0;
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

double WarmupMs(const std::vector<double>& warmup_samples,
                double steady_median) {
    double sum = 0.0;
    for (double ms : warmup_samples) {
        sum += ms;
    }
    const double overhead =
            sum - steady_median * static_cast<double>(warmup_samples.size());
    return overhead > 0.0 ? overhead : 0.0;
}

BenchRecord::BenchRecord(const std::string& example,
                         const std::string& device) {
#ifdef __ANDROID__
    const char* platform = "android";
#elif defined(_WIN32)
    const char* platform = "windows";
#else
    const char* platform = "linux";
#endif
    m_json = fmt::format(
            "\"example\":\"{}\",\"device\":\"{}\",\"platform\":\"{}\"",
            example, device, platform);
    add("os", OsString());
    add("cpu", CpuString());
    add("cpu_cores", int64_t(std::thread::hardware_concurrency()));
    add("ram_gb", RamGb(), 1);
}

void BenchRecord::add(const std::string& key, const std::string& value) {
    m_json += fmt::format(",\"{}\":\"{}\"", key, value);
}

void BenchRecord::add(const std::string& key, double value, int precision) {
    m_json += fmt::format(",\"{}\":{:.{}f}", key, value, precision);
}

void BenchRecord::add(const std::string& key, int64_t value) {
    m_json += fmt::format(",\"{}\":{}", key, value);
}

void BenchRecord::add(const std::string& key, bool value) {
    m_json += fmt::format(",\"{}\":{}", key, value ? "true" : "false");
}

void BenchRecord::addPacking(int64_t funcs, int64_t substages,
                             int64_t stages) {
    add("funcs", funcs);
    add("substages", substages);
    add("stages", stages);
}

void BenchRecord::save(const std::string& path) const {
    std::ofstream f(path);
    f << "{" << m_json << "}\n";
}

}  // namespace dressi_examples
