#include "factory/FilterFactory.h"

#include <stdexcept>

// -----------------------------
// Singleton Instance
// -----------------------------

FilterFactory& FilterFactory::instance() {
    static FilterFactory instance;
    return instance;
}

// -----------------------------
// Registration
// -----------------------------

void FilterFactory::registerCPU(
    const std::string& name,
    Creator creator
) {
    cpuRegistry[name] = creator;
}

void FilterFactory::registerCUDA(
    const std::string& name,
    Creator creator
) {
    cudaRegistry[name] = creator;
}

// -----------------------------
// Create Filter
// -----------------------------

std::unique_ptr<IFilter> FilterFactory::create(
    const std::string& name,
    const std::string& mode
) {
    if (mode == "cpu") {
        auto it = cpuRegistry.find(name);
        if (it != cpuRegistry.end()) {
            return it->second();
        }
    }
    else if (mode == "cuda") {
        auto it = cudaRegistry.find(name);
        if (it != cudaRegistry.end()) {
            return it->second();
        }
    }

    throw std::runtime_error(
        "Filter not found: " + name + " (mode: " + mode + ")"
    );
}

// -----------------------------
// Available Filters
// -----------------------------

std::vector<std::string> FilterFactory::getAvailableCPU() const {
    std::vector<std::string> result;
    result.reserve(cpuRegistry.size());

    for (const auto& pair : cpuRegistry) {
        result.push_back(pair.first);
    }

    return result;
}

std::vector<std::string> FilterFactory::getAvailableCUDA() const {
    std::vector<std::string> result;
    result.reserve(cudaRegistry.size());

    for (const auto& pair : cudaRegistry) {
        result.push_back(pair.first);
    }

    return result;
}