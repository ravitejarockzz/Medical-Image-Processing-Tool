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

std::unique_ptr<IFilter> FilterFactory::createCPU(
    const std::string& name
) {
        auto it = cpuRegistry.find(name);
        if (it != cpuRegistry.end()) {
            return it->second();
        }
    throw std::runtime_error(
        "Filter not found: " + name + " (mode: cpu)"
    );
}

std::unique_ptr<IFilter> FilterFactory::createCUDA(
    const std::string& name
) {
        auto it = cudaRegistry.find(name);
        if (it != cudaRegistry.end()) {
            return it->second();
        }

    throw std::runtime_error(
        "Filter not found: " + name + " (mode: cuda)"
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