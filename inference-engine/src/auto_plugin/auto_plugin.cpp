// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <unordered_map>
#include <unordered_set>


#include <ie_metric_helpers.hpp>
#include <auto_plugin/auto_config.hpp>
#include <ie_core.hpp>
#include <threading/ie_executor_manager.hpp>
#include <ie_algorithm.hpp>

#include "auto_plugin.hpp"

// ------------------------------AutoInferencePlugin----------------------------
namespace AutoPlugin {
    using namespace InferenceEngine;
namespace {
    std::map<std::string, std::string> mergeConfigs(std::map<std::string, std::string> config,
                                                    const std::map<std::string, std::string> & local) {
        for (auto && kvp : local) {
            config[kvp.first] = kvp.second;
        }
        return config;
    }
}  // namespace

AutoInferencePlugin::AutoInferencePlugin() {
    _pluginName = "AUTO";
    RegisterPolicy(SelectDevicePolicy::STATIC);
}

ExecutableNetworkInternal::Ptr AutoInferencePlugin::LoadExeNetworkImpl(const CNNNetwork &network,
                                                                       const std::map<std::string, std::string>& config) {
    if (GetCore() == nullptr) {
        IE_THROW() << "Please, work with AUTO device via InferencEngine::Core object";
    }

    if (network.getFunction() == nullptr) {
        IE_THROW() << "AUTO device supports just ngraph network representation";
    }

    auto fullConfig = mergeConfigs(_config, config);
    auto deviceChoiceConfig = fullConfig.find(AutoConfigParams::KEY_AUTO_DEVICE_CHOICE);
    if (deviceChoiceConfig == fullConfig.end()) {
        auto deviceChoice = GetDeviceChoice();
        fullConfig.emplace(AutoConfigParams::KEY_AUTO_DEVICE_CHOICE, deviceChoice);
    }
    auto metaDevices = ParseMetaDevices(fullConfig[AutoConfigParams::KEY_AUTO_DEVICE_CHOICE], fullConfig);

    // collect the settings that are applicable to the devices we are loading the network to
    std::unordered_map<std::string, InferenceEngine::Parameter> autoNetworkConfig;
    autoNetworkConfig.insert(*fullConfig.find(AutoConfigParams::KEY_AUTO_DEVICE_CHOICE));

    auto scheduleType = ParseScheduleType(fullConfig[AutoConfigParams::KEY_AUTO_SCHEDULE_TYPE]);
    auto optCap = GetOptimizationCapabilities();

    ExecutableNetwork executableNetwork;
    DeviceInformation selectedDevice {};
    while (!metaDevices.empty()) {
        selectedDevice = _policies[scheduleType]->SelectDevice(network, metaDevices, optCap);
        try {
            auto deviceQr = GetCore()->QueryNetwork(network, selectedDevice.deviceName, selectedDevice.config);
            executableNetwork = GetCore()->LoadNetwork(network, selectedDevice.deviceName, selectedDevice.config);
            // need record selected device config, sample like MULTI4.cpp
            autoNetworkConfig.insert(selectedDevice.config.begin(), selectedDevice.config.end());
            break;
        } catch(const InferenceEngine::Exception &iie) {
            std::cout << "[AUTO] LoadNetwork failed on device named "
                      << selectedDevice.deviceName << " with exception "
                      << iie.what() << std::endl;

            auto eraseDevice = std::find_if(metaDevices.begin(), metaDevices.end(),
                [=](const DeviceInformation& d)->bool{return d.deviceName == selectedDevice.deviceName;});
            if (eraseDevice == metaDevices.end()) {
                IE_THROW() << "Didn't find the selected device name";
            }
            metaDevices.erase(eraseDevice);
            selectedDevice = {};
        }
    }
    if (selectedDevice.deviceName.empty()) {
        IE_THROW(NotFound) << "Failed to load network to any device "
                           <<  "that the AUTO device is initialized to work with";
    }

    std::cout << "[AUTO] LoadNetwork schedule to device named "<< selectedDevice.deviceName << std::endl;

    bool enablePerfCounters = false;
    try {
        enablePerfCounters =
            executableNetwork.GetConfig(PluginConfigParams::KEY_PERF_COUNT).as<std::string>() ==
            PluginConfigParams::YES;
    } catch (...) {
    }

    return std::make_shared<AutoExecutableNetwork>(executableNetwork,
                                                   selectedDevice,
                                                   autoNetworkConfig,
                                                   enablePerfCounters);
}

QueryNetworkResult AutoInferencePlugin::QueryNetwork(const CNNNetwork&                         network,
                                                            const std::map<std::string, std::string>& config) const {
    QueryNetworkResult queryResult;
    if (GetCore() == nullptr) {
        IE_THROW() << "Please, work with AUTO device via InferencEngine::Core object";
    }

    if (network.getFunction() == nullptr) {
        IE_THROW() << "AUTO device supports just ngraph network representation";
    }

    queryResult.rc = StatusCode::OK;
    queryResult.supportedLayersMap.clear();

    auto fullConfig = mergeConfigs(_config, config);
    auto deviceChoiceItr = fullConfig.find(AutoConfigParams::KEY_AUTO_DEVICE_CHOICE);
    if (deviceChoiceItr == fullConfig.end()) {
        auto deviceChoice = GetDeviceChoice();
        fullConfig.emplace(AutoConfigParams::KEY_AUTO_DEVICE_CHOICE, deviceChoice);
    }
    auto metaDevices = ParseMetaDevices(fullConfig[AutoConfigParams::KEY_AUTO_DEVICE_CHOICE], fullConfig);
    std::unordered_set<std::string> supportedLayers;
    std::unordered_set<std::string> supportedDevices;
    for (auto&& value : metaDevices) {
        try {
            if (value.deviceName != _pluginName) {
                auto deviceQr = GetCore()->QueryNetwork(network, value.deviceName,
                                                        value.config);
                std::unordered_set<std::string> deviceSupportedLayers;
                for (auto &&layerQr : deviceQr.supportedLayersMap) {
                    deviceSupportedLayers.emplace(layerQr.first);
                }
                supportedLayers = supportedLayers.empty()
                                ? deviceSupportedLayers : (deviceSupportedLayers.empty()
                                ? supportedLayers : InferenceEngine::details::Intersection(
                                     supportedLayers, deviceSupportedLayers));
                supportedDevices.insert(value.deviceName);
            }
        } catch (...) {
            std::cout << "[AUTO] " << value.deviceName << " doesn't support QueryNetwork\n";
        }
    }

    if (supportedDevices.empty()) {
        IE_THROW() << "Please, check environment due to no supported devices can be used";
    }
    std::cout << "[AUTO] The below devices support QueryNetwork: ";
    std::copy(supportedDevices.begin(), supportedDevices.end(), std::ostream_iterator<std::string>(std::cout, " "));
    std::cout << std::endl;

    for (auto&& supportedLayer : supportedLayers) {
        queryResult.supportedLayersMap[supportedLayer] = GetName();
    }
    return queryResult;
}

InferenceEngine::Parameter AutoInferencePlugin::GetConfig(const std::string& name,
                                                          const std::map<std::string, InferenceEngine::Parameter> & options) const {
    auto it = _config.find(name);
    if (it == _config.end()) {
        return {};
    } else {
        return { it->second };
    }
}

void AutoInferencePlugin::SetConfig(const std::map<std::string, std::string> & config) {
    for (auto && kvp : config) {
        _config[kvp.first] = kvp.second;
    }
}

InferenceEngine::Parameter AutoInferencePlugin::GetMetric(const std::string& name,
                                                          const std::map<std::string, InferenceEngine::Parameter> & options) const {
    if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        std::vector<std::string> metrics;
        metrics.push_back(METRIC_KEY(AVAILABLE_DEVICES));
        metrics.push_back(METRIC_KEY(SUPPORTED_METRICS));
        metrics.push_back(METRIC_KEY(FULL_DEVICE_NAME));
        metrics.push_back(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
        metrics.push_back(METRIC_KEY(OPTIMIZATION_CAPABILITIES));
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, metrics);
    } else if (name == METRIC_KEY(AVAILABLE_DEVICES)) {
        // FIXME: available devices should get from thirdparty
        std::vector<std::string> availableDevices = { "" };
        IE_SET_METRIC_RETURN(AVAILABLE_DEVICES, availableDevices);
    } else if (name == METRIC_KEY(FULL_DEVICE_NAME)) {
        std::string device_name = { "AUTO" };
        IE_SET_METRIC_RETURN(FULL_DEVICE_NAME, device_name);
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        std::vector<std::string> configKeys = {
                // AutoConfigParams::KEY_AUTO_DEVICE_CHOICE,
                AutoConfigParams::KEY_AUTO_SCHEDULE_TYPE
        };
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, configKeys);
    } else if (name == METRIC_KEY(OPTIMIZATION_CAPABILITIES)) {
        std::vector<std::string> capabilities = GetOptimizationCapabilities();
        IE_SET_METRIC_RETURN(OPTIMIZATION_CAPABILITIES, capabilities);
    } else {
        IE_THROW() << "Unsupported metric key " << name;
    }
}

std::string AutoInferencePlugin::GetDeviceChoice() {
    // TODO: should delete this WA after thirdparty support query devices.
    Core ie;
    auto availableDevices = ie.GetAvailableDevices();

    if (availableDevices.empty()) {
        IE_THROW() << "No available devices";
    }

    std::string allDevices;
    for (auto && device : availableDevices) {
        if (device == "AUTO") {
            continue;
        }
        allDevices += device;
        allDevices += ((device == availableDevices[availableDevices.size()-1]) ? "" : ",");
    }
    std::cout << "[AUTO] Available device lists: " << allDevices << std::endl;

    return allDevices;
}

SelectDevicePolicy AutoInferencePlugin::ParseScheduleType(const std::string & scheduleType) {
    if (scheduleType == "STATIC" || scheduleType.empty()) {
        return SelectDevicePolicy::STATIC;
    }
    IE_THROW(NotImplemented) << "Auto plugin doesn't implement schedule method with type " << scheduleType;
}

//////////////////////////////////// private & protected functions ///////////////////
void AutoInferencePlugin::RegisterPolicy(SelectDevicePolicy type) {
    _policies.emplace(type, new AutoSelectDevice(type));
}

std::vector<std::string> AutoInferencePlugin::GetOptimizationCapabilities() const {
    // FIXME: workaround to get devicelist.
    std::vector<std::string> capabilities;
    std::vector<std::string> queryDeviceLists{"CPU", "GPU", "GNA", "VPUX"};
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

std::map<std::string, std::string> AutoInferencePlugin::GetSupportedConfig(
        const std::map<std::string, std::string> & config, const std::string & deviceName) const {
    std::vector<std::string> supportedConfigKeys = GetCore()->GetMetric(deviceName, METRIC_KEY(SUPPORTED_CONFIG_KEYS));
    std::map<std::string, std::string> supportedConfig;
    for (auto&& key : supportedConfigKeys) {
        auto itKey = config.find(key);
        if (config.end() != itKey) {
            supportedConfig[key] = itKey->second;
        }
    }
    return supportedConfig;
}

std::vector<DeviceInformation> AutoInferencePlugin::ParseMetaDevices(const std::string& deviceChoice,
                                                                     const std::map<std::string, std::string> & config) const {
    std::vector<DeviceInformation> metaDevices;
    // parsing the string and splitting to tokens
    std::vector<std::string> devicesWithRequests;
    // parsing the string and splitting the comma-separated tokens
    std::string::size_type i = 0;
    std::string::size_type idelimeter;
    while ((idelimeter = deviceChoice.find(',', i)) != std::string::npos) {
        devicesWithRequests.push_back(deviceChoice.substr(i, idelimeter - i));
        i = idelimeter + 1;
    }
    // last token in the string (which has no comma after that)
    devicesWithRequests.push_back(deviceChoice.substr(i, deviceChoice.length() - i));

    auto getDeviceConfig = [&] (const DeviceName & deviceWithID) {
        DeviceIDParser deviceParser(deviceWithID);
        std::string deviceName = deviceParser.getDeviceName();
        std::map<std::string, std::string> tconfig = mergeConfigs(_config, config);

        // set device ID if any
        std::string deviceIDLocal = deviceParser.getDeviceID();
        if (!deviceIDLocal.empty()) {
            tconfig[PluginConfigParams::KEY_DEVICE_ID] = deviceIDLocal;
        }

        return GetSupportedConfig(tconfig, deviceName);
    };

    for (auto && d : devicesWithRequests) {
        // create meta device
        metaDevices.push_back({ d, getDeviceConfig(d)});
    }

    return metaDevices;
}

// define CreatePluginEngine to create plugin instance
static const Version version = {{2, 1}, CI_BUILD_NUMBER, "AutoPlugin"};
IE_DEFINE_PLUGIN_CREATE_FUNCTION(AutoInferencePlugin, version)
}  // namespace AutoPlugin
