// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "auto_schedule_policy.hpp"

namespace AutoPlugin {

static void printInputAndOutputsInfo(const InferenceEngine::CNNNetwork& network) {
    std::cout << "Network inputs:" << std::endl;
    for (auto&& layer : network.getInputsInfo()) {
        std::cout << "    " << layer.first << " : " << layer.second->getPrecision() << " / " << layer.second->getLayout() << std::endl;
    }
    std::cout << "Network outputs:" << std::endl;
    for (auto&& layer : network.getOutputsInfo()) {
        std::cout << "    " << layer.first << " : " << layer.second->getPrecision() << " / " << layer.second->getLayout() << std::endl;
    }
}

class AutoSchedulePolicy::Priv{
public:
    virtual ~Priv() = default;
    virtual VecDeviceCiter
    SelectDevice(const InferenceEngine::CNNNetwork &network, const VecDevice& metaDevices) const = 0;
};

class AutoStaticPolicy: public AutoSchedulePolicy::Priv{
public:
    VecDeviceCiter SelectDevice(const InferenceEngine::CNNNetwork &network, const VecDevice& metaDevices) const override;

private:
    mutable std::mutex _mutex;
};

VecDeviceCiter AutoStaticPolicy::SelectDevice(const InferenceEngine::CNNNetwork &network, const VecDevice& metaDevices) const {
    printInputAndOutputsInfo(network);
    // 1. GPU is an alias for GPU.0
    // 2. GPU.0 is always iGPU if system has iGPU
    // 3. GPU.X where X={1,2,3,...} is dGPU if system has both iGPU and dGPU
    // 4. GPU.0 could be dGPU if system has no iGPU
    static VecDevice VPUX;
    static VecDevice GPU;
    static VecDevice GNA;
    static VecDevice CPU;
    std::lock_guard<std::mutex> lockGuard(_mutex);
    VPUX.clear();
    GPU.clear();
    GNA.clear();
    CPU.clear();
    for (auto& item : metaDevices) {
        if (item.deviceName.find("VPUX") == 0) {
            VPUX.push_back(item);
            continue;
        }
        if (item.deviceName.find("GPU") == 0) {
            GPU.push_back(item);
            continue;
        }
        if (item.deviceName.find("GNA") == 0) {
            GNA.push_back(item);
            continue;
        }
        if (item.deviceName.find("CPU") == 0) {
            CPU.push_back(item);
            continue;
        }
        IE_THROW(NotImplemented) << "Auto plugin doesn't support device named " << item.deviceName;
    }
    if (VPUX.empty() && GPU.empty() && GNA.empty() && CPU.empty()) {
        IE_THROW(NotFound) << "No available device found";
    }
    std::sort(GPU.begin(), GPU.end(), [](DeviceInformation& a, DeviceInformation& b)->bool{return b.deviceName < a.deviceName;});

    return !VPUX.empty()
           ? VPUX.begin(): !GPU.empty()
           ? GPU.begin() : !GNA.empty()
           ? GNA.begin() : CPU.begin();
}

AutoSchedulePolicy::AutoSchedulePolicy(SchedulePolicyType type) {
    switch (type) {
        case SchedulePolicyType::STATIC: {
            _priv.reset(new AutoStaticPolicy());
            break;
        }
        default: {
            IE_THROW(NotImplemented)
                << "Does not implement schedule type " << StrPolicy(type);
        }
    }
}

AutoSchedulePolicy::~AutoSchedulePolicy() = default;

VecDeviceCiter AutoSchedulePolicy::SelectDevice(const InferenceEngine::CNNNetwork &network, const VecDevice& metaDevices) const {
    return _priv->SelectDevice(network, metaDevices);
}

std::string AutoSchedulePolicy::StrPolicy(SchedulePolicyType type) {
    switch (type) {
    case SchedulePolicyType::STATIC:
        return "STATIC";
    default:
        return "UNSUPPORT";
    }
}
} // namespace AutoPlugin