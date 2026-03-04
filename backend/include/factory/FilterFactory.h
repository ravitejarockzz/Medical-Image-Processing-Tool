#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <functional>

#include "core/IFilter.h"

class FilterFactory {
public:
    using Creator = std::function<std::unique_ptr<IFilter>()>;

    // Singleton access
    static FilterFactory& instance();

    // Register CPU filter
    void registerCPU(const std::string& name, Creator creator);

    // Register CUDA filter
    void registerCUDA(const std::string& name, Creator creator);

    // Create filter by name + mode
    std::unique_ptr<IFilter> create(
        const std::string& name,
        const std::string& mode
    );

    // List available operations
    std::vector<std::string> getAvailableCPU() const;
    std::vector<std::string> getAvailableCUDA() const;

private:
    FilterFactory() = default;

    std::unordered_map<std::string, Creator> cpuRegistry;
    std::unordered_map<std::string, Creator> cudaRegistry;
};

//
// 🔥 Auto-Registration Macros
//

#define REGISTER_CPU_FILTER(ClassType, NameString)          \
    namespace {                                             \
        const bool registered_##ClassType##_cpu = []() {    \
            FilterFactory::instance().registerCPU(          \
                NameString,                                 \
                []() { return std::make_unique<ClassType>(); } \
            );                                              \
            return true;                                    \
        }();                                                \
    }

#define REGISTER_CUDA_FILTER(ClassType, NameString)         \
    namespace {                                             \
        const bool registered_##ClassType##_cuda = []() {   \
            FilterFactory::instance().registerCUDA(         \
                NameString,                                 \
                []() { return std::make_unique<ClassType>(); } \
            );                                              \
            return true;                                    \
        }();                                                \
    }