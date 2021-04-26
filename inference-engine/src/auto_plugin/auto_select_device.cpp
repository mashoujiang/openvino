// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "auto_select_device.hpp"
#include <ie_precision.hpp>

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

class AutoSelectDevice::Priv{
public:
    virtual ~Priv() = default;
    virtual DeviceInformation SelectDevice(const InferenceEngine::CNNNetwork &network,
                                           const VecDevice& metaDevices,
                                           const std::vector<std::string>& optCap) const = 0;
};

class AutoStaticPolicy: public AutoSelectDevice::Priv{
public:
    DeviceInformation SelectDevice(const InferenceEngine::CNNNetwork &network,
                                   const VecDevice& metaDevices,
                                   const std::vector<std::string>& optCap) const override;

private:
    static std::string  GetNetworkPrecision(const InferenceEngine::CNNNetwork &network);
private:
    mutable std::mutex _mutex;
};

DeviceInformation AutoStaticPolicy::SelectDevice(const InferenceEngine::CNNNetwork &network,
                                                 const VecDevice& metaDevices,
                                                 const std::vector<std::string>& optCap) const {
    printInputAndOutputsInfo(network);
    // 1. GPU is an alias for GPU.0
    // 2. GPU.0 is always iGPU if system has iGPU
    // 3. GPU.X where X={1,2,3,...} is dGPU if system has both iGPU and dGPU
    // 4. GPU.0 could be dGPU if system has no iGPU
    VecDevice VPUX;
    VecDevice GPU;
    VecDevice GNA;
    VecDevice CPU;
    std::lock_guard<std::mutex> lockGuard(_mutex);
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
    // dGPU is preferred
    std::sort(GPU.begin(), GPU.end(), [](DeviceInformation& a, DeviceInformation& b)->bool{return b.deviceName < a.deviceName;});

    if (!VPUX.empty()) {
        return VPUX[0];
    }

    if (!GPU.empty()) {
        return GPU[0];
    }

    if (!GNA.empty()) {
        return GNA[0];
    }

    if (CPU.empty()) {
        IE_THROW(NotFound) << "No available device could be used";
    }
    return CPU[0];
}

std::string AutoStaticPolicy::GetNetworkPrecision(const InferenceEngine::CNNNetwork &network) {
  for (auto&& layer : network.getInputsInfo()) {
      auto precision = layer.second->getPrecision();
      auto name = std::string(precision.name());
      // FIXME: presentation mismatch between device precision and network precision
      if (name == "I8") {
          name = "INT8";
      }
      // FIXME: only choose the first layer precision
      return name;
  }
  return {};
}

AutoSelectDevice::AutoSelectDevice(SelectDevicePolicy type) {
    switch (type) {
        case SelectDevicePolicy::STATIC: {
            _priv.reset(new AutoStaticPolicy());
            break;
        }
        default: {
            IE_THROW(NotImplemented)
                << "Does not implement select device policy with type " << StrPolicy(type);
        }
    }
}

AutoSelectDevice::~AutoSelectDevice() = default;

DeviceInformation AutoSelectDevice::SelectDevice(const InferenceEngine::CNNNetwork &network,
                                                   const VecDevice& metaDevices,
                                                   const std::vector<std::string>& optCap) const {
    return _priv->SelectDevice(network, metaDevices, optCap);
}

std::string AutoSelectDevice::StrPolicy(SelectDevicePolicy type) {
    switch (type) {
    case SelectDevicePolicy::STATIC:
        return "STATIC";
    default:
        return "UNSUPPORT";
    }
}
} // namespace AutoPlugin