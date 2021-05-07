// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <unordered_set>

#include <ie_metric_helpers.hpp>
#include <ie_core.hpp>
#include <threading/ie_executor_manager.hpp>
#include <ie_algorithm.hpp>

#include "auto_plugin.hpp"

namespace AutoPlugin {
namespace {
    ConfigType mergeConfigs(ConfigType config, const ConfigType& local) {
        for (auto && kvp : local) {
            config[kvp.first] = kvp.second;
        }
        return config;
    }

    std::string GetNetworkPrecision(const InferenceEngine::CNNNetwork &network) {
        for (auto&& layer : network.getInputsInfo()) {
            auto precision = layer.second->getPrecision();
            auto name = std::string(precision.name());
            if (name == "I8") {
                name = "INT8";
            }
            return name;
        }
        return {};
    }

    DeviceInformation SelectDevice(const InferenceEngine::CNNNetwork &network,
                                   const std::vector<DeviceInformation>& metaDevices,
                                   const std::vector<std::string>& optCap) {
        std::vector<DeviceInformation> VPUX;
        std::vector<DeviceInformation> GPU;
        std::vector<DeviceInformation> GNA;
        std::vector<DeviceInformation> CPU;
        std::vector<DeviceInformation> MYRIAD;
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
            if (item.deviceName.find("MYRIAD") == 0) {
                MYRIAD.push_back(item);
                continue;
            }
            if (item.deviceName.find("CPU") == 0) {
                CPU.push_back(item);
                continue;
            }
            IE_THROW(NotImplemented) << "Auto plugin doesn't support device named " << item.deviceName;
        }

        // Get network precision
        auto networkPrecision = GetNetworkPrecision(network);

        auto getCap = [&](std::string&& substr){
            auto capability = std::find_if(optCap.begin(), optCap.end(),
                                        [&](const std::string& c)->bool{ return (c.find(substr) != std::string::npos);});
            return capability;
        };

        if (VPUX.empty() && GPU.empty() && GNA.empty() && MYRIAD.empty() && CPU.empty()) {
            IE_THROW(NotFound) << "No available device found";
        }
        // dGPU is preferred
        std::sort(GPU.begin(), GPU.end(), [](DeviceInformation& a, DeviceInformation& b)->bool{return b.deviceName < a.deviceName;});

        if (!VPUX.empty()) {
            auto capability = getCap("VPUX");
            if (capability != optCap.end() && capability->find(networkPrecision) != std::string::npos) {
            return VPUX[0];
            }
        }

        if (!GPU.empty()) {
            auto capability = getCap("GPU");
            if (capability != optCap.end() && capability->find(networkPrecision) != std::string::npos) {
                return GPU[0];
            }
        }

        if (!GNA.empty()) {
            auto capability = getCap("GNA");
            if (capability != optCap.end() && capability->find(networkPrecision) != std::string::npos) {
                return GNA[0];
            }
        }

        if (!MYRIAD.empty()) {
            auto capability = getCap("MYRIAD");
            if (capability != optCap.end() && capability->find(networkPrecision) != std::string::npos) {
                return MYRIAD[0];
            }
        }

        if (CPU.empty()) {
            IE_THROW(NotFound) << "No available device could be used";
        }
        return CPU[0];
    }
} // namespace

AutoInferencePlugin::AutoInferencePlugin() {
    _pluginName = "AUTO";
}

std::vector<std::string> AutoInferencePlugin::GetOptimizationCapabilities() const {
    std::vector<std::string> capabilities;
    std::vector<std::string> queryDeviceLists{"CPU", "GPU", "GNA", "MYRIAD", "VPUX"};
    for (auto& item : queryDeviceLists) {
        try {
            std::vector<std::string> device_cap =
                    GetCore()->GetMetric(item, METRIC_KEY(OPTIMIZATION_CAPABILITIES));
            std::string device_cap_str = item + ": ";
            for (auto &dc : device_cap) {
                device_cap_str += dc + " ";
            }
            capabilities.push_back(device_cap_str);
        } catch (...) {
        }
    }
    return capabilities;
}

IE::ExecutableNetworkInternal::Ptr AutoInferencePlugin::LoadExeNetworkImpl(const IE::CNNNetwork& network, const ConfigType& config) {
    if (GetCore() == nullptr) {
        IE_THROW() << "Please, work with AUTO device via InferencEngine::Core object";
    }

    if (network.getFunction() == nullptr) {
        IE_THROW() << "AUTO device supports just ngraph network representation";
    }

    auto fullConfig = mergeConfigs(_config, config);
    auto deviceChoice = GetDeviceChoice();
    auto metaDevices = ParseMetaDevices(deviceChoice, fullConfig);
    auto optCap = GetOptimizationCapabilities();

    DeviceInformation selectedDevice = SelectDevice(network, metaDevices, optCap);
    IE::ExecutableNetwork executableNetwork;
    try {
        auto deviceQr = GetCore()->QueryNetwork(network, selectedDevice.deviceName, selectedDevice.config);
        executableNetwork = GetCore()->LoadNetwork(network, selectedDevice.deviceName, selectedDevice.config);
    } catch(const IE::Exception &iie) {
        IE_THROW() << "Failed to load network to device named " << selectedDevice.deviceName;
    }

    bool enablePerfCounters = false;
    try {
        enablePerfCounters =
            executableNetwork.GetConfig(IE::PluginConfigParams::KEY_PERF_COUNT).as<std::string>() ==
                IE::PluginConfigParams::YES;
    } catch (...) {
    }

    return std::make_shared<AutoExecutableNetwork>(executableNetwork,
                                                   selectedDevice,
                                                   enablePerfCounters);
}

IE::QueryNetworkResult AutoInferencePlugin::QueryNetwork(const IE::CNNNetwork& network, const ConfigType& config) const {
    IE::QueryNetworkResult queryResult;
    if (GetCore() == nullptr) {
        IE_THROW() << "Please, work with AUTO device via InferencEngine::Core object";
    }

    if (network.getFunction() == nullptr) {
        IE_THROW() << "AUTO device supports just ngraph network representation";
    }

    queryResult.rc = IE::StatusCode::OK;
    queryResult.supportedLayersMap.clear();

    auto fullConfig = mergeConfigs(_config, config);
    auto deviceChoice = GetDeviceChoice();
    auto metaDevices = ParseMetaDevices(deviceChoice, fullConfig);
    std::unordered_set<std::string> supportedLayers;
    std::unordered_set<std::string> supportedDevices;
    for (auto&& value : metaDevices) {
        try {
            if (value.deviceName != _pluginName) {
                auto deviceQr = GetCore()->QueryNetwork(network, value.deviceName, value.config);
                std::unordered_set<std::string> deviceSupportedLayers;
                for (auto &&layerQr : deviceQr.supportedLayersMap) {
                    deviceSupportedLayers.emplace(layerQr.first);
                }
                supportedLayers = supportedLayers.empty()
                                ? deviceSupportedLayers : (deviceSupportedLayers.empty()
                                ? supportedLayers : IE::details::Intersection(
                                     supportedLayers, deviceSupportedLayers));
                supportedDevices.insert(value.deviceName);
            }
        } catch (...) {
        }
    }

    if (supportedDevices.empty()) {
        IE_THROW() << "Please, check environment due to no supported devices can be used";
    }

    for (auto&& supportedLayer : supportedLayers) {
        queryResult.supportedLayersMap[supportedLayer] = GetName();
    }
    return queryResult;
}

IE::Parameter AutoInferencePlugin::GetConfig(const std::string& name,
                                             const std::map<std::string, IE::Parameter> & options) const {
    auto it = _config.find(name);
    if (it == _config.end()) {
        IE_THROW() << "Unsupported config key: " << name;
    } else {
        return { it->second };
    }
}

void AutoInferencePlugin::SetConfig(const ConfigType& config) {
    for (auto && kvp : config) {
        _config[kvp.first] = kvp.second;
    }
}

IE::Parameter AutoInferencePlugin::GetMetric(const std::string& name,
                                             const std::map<std::string, IE::Parameter> & options) const {
    if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        std::vector<std::string> metrics;
        metrics.emplace_back(METRIC_KEY(AVAILABLE_DEVICES));
        metrics.emplace_back(METRIC_KEY(SUPPORTED_METRICS));
        metrics.emplace_back(METRIC_KEY(FULL_DEVICE_NAME));
        metrics.emplace_back(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
        metrics.emplace_back(METRIC_KEY(OPTIMIZATION_CAPABILITIES));
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, metrics);
    } else if (name == METRIC_KEY(AVAILABLE_DEVICES)) {
        // FIXME: available devices should get from thirdparty
        std::vector<std::string> availableDevices = { "" };
        IE_SET_METRIC_RETURN(AVAILABLE_DEVICES, availableDevices);
    } else if (name == METRIC_KEY(FULL_DEVICE_NAME)) {
        std::string device_name = { "AUTO" };
        IE_SET_METRIC_RETURN(FULL_DEVICE_NAME, device_name);
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        std::vector<std::string> configKeys;
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, configKeys);
    } else if (name == METRIC_KEY(OPTIMIZATION_CAPABILITIES)) {
        std::vector<std::string> capabilities = { "" };
        IE_SET_METRIC_RETURN(OPTIMIZATION_CAPABILITIES, capabilities);
    } else {
        IE_THROW() << "Unsupported metric key " << name;
    }
}

std::string AutoInferencePlugin::GetDeviceChoice() const {
    std::vector<std::string> availableDevices = GetCore()->GetAvailableDevices();
    std::string allDevices;
    for (auto && device : availableDevices) {
        if (device == "AUTO") {
            continue;
        }
        allDevices += device;
        allDevices += ((device == availableDevices[availableDevices.size()-1]) ? "" : ",");
    }

    return allDevices;
}

//////////////////////////////////// private & protected functions ///////////////////
ConfigType AutoInferencePlugin::GetSupportedConfig(const ConfigType&  config,
                                                   const std::string& deviceName) const {
    std::vector<std::string> supportedConfigKeys = GetCore()->GetMetric(deviceName, METRIC_KEY(SUPPORTED_CONFIG_KEYS));
    ConfigType supportedConfig;
    for (auto&& key : supportedConfigKeys) {
        auto itKey = config.find(key);
        if (config.end() != itKey) {
            supportedConfig[key] = itKey->second;
        }
    }
    return supportedConfig;
}

std::vector<DeviceInformation> AutoInferencePlugin::ParseMetaDevices(const std::string& deviceChoice,
                                                                     const ConfigType&  config) const {
    std::vector<DeviceInformation> metaDevices;
    // Parsing the string and splitting to tokens
    std::vector<std::string> devicesWithRequests;
    // Parsing the string and splitting the comma-separated tokens
    std::string::size_type i = 0;
    std::string::size_type idelimeter;
    while ((idelimeter = deviceChoice.find(',', i)) != std::string::npos) {
        devicesWithRequests.push_back(deviceChoice.substr(i, idelimeter - i));
        i = idelimeter + 1;
    }
    // Last token in the string (which has no comma after that)
    devicesWithRequests.push_back(deviceChoice.substr(i, deviceChoice.length() - i));

    auto getDeviceConfig = [&] (const DeviceName & deviceWithID) {
        IE::DeviceIDParser deviceParser(deviceWithID);
        std::string deviceName = deviceParser.getDeviceName();
        ConfigType tconfig = mergeConfigs(_config, config);

        // Set device ID if any
        std::string deviceIDLocal = deviceParser.getDeviceID();
        if (!deviceIDLocal.empty()) {
            tconfig[IE::PluginConfigParams::KEY_DEVICE_ID] = deviceIDLocal;
        }

        return GetSupportedConfig(tconfig, deviceName);
    };

    for (auto && d : devicesWithRequests) {
        // Create meta device
        metaDevices.push_back({ d, getDeviceConfig(d)});
    }

    return metaDevices;
}

// Define CreatePluginEngine to create plugin instance
static const IE::Version version = {{2, 1}, CI_BUILD_NUMBER, "AutoPlugin"};
IE_DEFINE_PLUGIN_CREATE_FUNCTION(AutoInferencePlugin, version)
}  // namespace AutoPlugin
