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

std::vector<DeviceInformation> AutoInferencePlugin::ParseMetaDevices(const std::string& priorities,
                                                                          const std::map<std::string, std::string> & config) const {
    std::vector<DeviceInformation> metaDevices;

    // parsing the string and splitting to tokens
    std::vector<std::string> devicesWithRequests;
    // parsing the string and splitting the comma-separated tokens
    std::string::size_type i = 0;
    std::string::size_type idelimeter;
    while ((idelimeter = priorities.find(',', i)) != std::string::npos) {
        devicesWithRequests.push_back(priorities.substr(i, idelimeter - i));
        i = idelimeter + 1;
    }
    // last token in the string (which has no comma after that)
    devicesWithRequests.push_back(priorities.substr(i, priorities.length() - i));

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
        auto openingBracket = d.find_first_of('(');
        auto closingBracket = d.find_first_of(')', openingBracket);
        auto deviceName = d.substr(0, openingBracket);

        int numRequests = -1;
        if (closingBracket != std::string::npos && openingBracket < closingBracket) {
            numRequests = std::stol(d.substr(openingBracket + 1, closingBracket - 1));

            if (numRequests <= 0) {
                IE_THROW() << "Priority value for '" << deviceName << "' must be > 0, while " << numRequests
                    << "is passed";
            }
        }

        // create meta device
        metaDevices.push_back({ deviceName, getDeviceConfig(deviceName), numRequests });
    }

    return metaDevices;
}

InferenceEngine::Parameter AutoInferencePlugin::GetConfig(const std::string& name,
        const std::map<std::string, InferenceEngine::Parameter> & options) const {
    auto it = _config.find(name);
    if (it == _config.end() && name != AutoConfigParams::KEY_AUTO_DEVICE_PRIORITIES) {
        IE_THROW() << "Unsupported config key: " << name << ", Or forgot to SetConfig";
    } else {
        return { it->second };
    }
}

void AutoInferencePlugin::SetConfig(const std::map<std::string, std::string> & config) {
    for (auto && kvp : config) {
        _config[kvp.first] = kvp.second;
    }
}

static const Version version = {{2, 1}, CI_BUILD_NUMBER, "AutoPlugin"};
IE_DEFINE_PLUGIN_CREATE_FUNCTION(AutoInferencePlugin, version)

AutoInferencePlugin::AutoInferencePlugin() {
    _pluginName = "AUTO";
}

InferenceEngine::Parameter AutoInferencePlugin::GetMetric(const std::string& name,
                                         const std::map<std::string, InferenceEngine::Parameter> & options) const {
    if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        std::vector<std::string> metrics;
        metrics.push_back(METRIC_KEY(SUPPORTED_METRICS));
        metrics.push_back(METRIC_KEY(FULL_DEVICE_NAME));
        metrics.push_back(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
        metrics.push_back(METRIC_KEY(OPTIMIZATION_CAPABILITIES));
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, metrics);
    } else if (name == METRIC_KEY(FULL_DEVICE_NAME)) {
        std::string device_name = { "AUTO" };
        IE_SET_METRIC_RETURN(FULL_DEVICE_NAME, device_name);
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        std::vector<std::string> configKeys = {
            AutoConfigParams::KEY_AUTO_DEVICE_PRIORITIES};
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, configKeys);
    } else if (name == METRIC_KEY(OPTIMIZATION_CAPABILITIES)) {
        std::vector<std::string> capabilities = GetOptimizationCapabilities();
        IE_SET_METRIC_RETURN(OPTIMIZATION_CAPABILITIES, capabilities);
    } else {
        IE_THROW() << "Unsupported metric key " << name;
    }
}

std::vector<std::string> AutoInferencePlugin::GetOptimizationCapabilities() const {
    std::vector<std::string> capabilities;
    std::vector<std::string> queryDeviceLists{"CPU", "GPU", "GNA", "VPU"};
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


std::string AutoInferencePlugin::GetPriorityDevices() {
    // TODO: should delete this WA after thirdparty support query devices.
    Core ie;
    auto availableDevices = ie.GetAvailableDevices();

    if (availableDevices.empty()) {
        IE_THROW() << "No available devices";
    }
    // sort devices: VPU > GNA > GPU > CPU
    std::sort(availableDevices.begin(), availableDevices.end(), [](std::string& a, std::string&b)->bool{ return b < a;});

    std::string allDevices;
    for (auto && device : availableDevices) {
        allDevices += device;
        allDevices += ((device == availableDevices[availableDevices.size()-1]) ? "" : ",");
    }
    std::cout << "Available device lists: " << allDevices << std::endl;

    return allDevices;
}

ExecutableNetworkInternal::Ptr AutoInferencePlugin::LoadExeNetworkImpl(const CNNNetwork &network,
                                                                              const std::map<std::string, std::string>& config) {
    if (GetCore() == nullptr) {
        IE_THROW() << "Please, work with AUTO device via InferencEngine::Core object";
    }

    if (network.getFunction() == nullptr) {
        IE_THROW() << "AUTO device supports just ngraph network representation";
    }

    if (_supportedDevices.empty()) {
       QueryNetwork(network, config);
    }

    auto fullConfig = mergeConfigs(_config, config);
    auto priorities = fullConfig.find(AutoConfigParams::KEY_AUTO_DEVICE_PRIORITIES);
    if (priorities == fullConfig.end()) {
        std::string priorityDevices;
        for (auto && device : _supportedDevices) {
            priorityDevices += device + ",";
        }
        priorityDevices.pop_back();

        std::cout << "KEY_AUTO_DEVICE_PRIORITIES key is not found, AUTO schedule to " << priorityDevices << std::endl;
        fullConfig.emplace(AutoConfigParams::KEY_AUTO_DEVICE_PRIORITIES, priorityDevices);
    }

    auto metaDevices = ParseMetaDevices(fullConfig[AutoConfigParams::KEY_AUTO_DEVICE_PRIORITIES], fullConfig);

    // collect the settings that are applicable to the devices we are loading the network to
    std::unordered_map<std::string, InferenceEngine::Parameter> autoNetworkConfig;
    autoNetworkConfig.insert(*fullConfig.find(AutoConfigParams::KEY_AUTO_DEVICE_PRIORITIES));

    DeviceMap<ExecutableNetwork> executableNetworkPerDevice;
    std::mutex load_mutex;
    std::vector<Task> loads;
    for (auto& p : metaDevices) {
        if (_supportedDevices.find(p.deviceName) != _supportedDevices.end()) {
            if (p.deviceName != _pluginName) {
                loads.push_back([&]() {
                    const auto &deviceName = p.deviceName;
                    const auto &deviceConfig = p.config;
                    auto exec_net =
                        GetCore()->LoadNetwork(network, deviceName, deviceConfig);
                    std::unique_lock<std::mutex> lock{load_mutex};
                    executableNetworkPerDevice.insert({deviceName, exec_net});
                    autoNetworkConfig.insert(deviceConfig.begin(),
                                             deviceConfig.end());
                });
            }
        }
    }
    auto executor = InferenceEngine::ExecutorManager::getInstance()->getIdleCPUStreamsExecutor(
            IStreamsExecutor::Config{"AutoAsyncLoad",
                                     static_cast<int>(std::thread::hardware_concurrency()) /* max possible #streams*/,
                                     1 /*single thread per stream*/,
                                     IStreamsExecutor::ThreadBindingType::NONE});
    executor->runAndWait(loads);
    if (executableNetworkPerDevice.empty())
        IE_THROW(NotFound) << "Failed to load network to any device "
                                            <<  "that the AUTO device is initialized to work with";

    // checking the perf counters config from the loaded network to respect both device's plugin and load-specific setting
    size_t num_plugins_supporting_perf_counters = 0;
    for (auto n : executableNetworkPerDevice) {
            try {
                num_plugins_supporting_perf_counters +=
                        n.second.GetConfig(PluginConfigParams::KEY_PERF_COUNT).as<std::string>() ==
                        PluginConfigParams::YES;
            } catch (...) {
            }
    }
    // AUTO can enable the perf counters only if all  devices support/enable that
    bool enablePerfCounters = num_plugins_supporting_perf_counters == executableNetworkPerDevice.size();
    return std::make_shared<AutoExecutableNetwork>(executableNetworkPerDevice,
                                                          metaDevices,
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
    auto priorities = fullConfig.find(AutoConfigParams::KEY_AUTO_DEVICE_PRIORITIES);
    if (priorities == fullConfig.end()) {
        auto priorityDevices = GetPriorityDevices();
        std::cout << "KEY_AUTO_DEVICE_PRIORITIES key is not found, AUTO schedule to " << priorityDevices << std::endl;
        fullConfig.emplace(AutoConfigParams::KEY_AUTO_DEVICE_PRIORITIES, priorityDevices);
    }
    auto metaDevices = ParseMetaDevices(fullConfig[AutoConfigParams::KEY_AUTO_DEVICE_PRIORITIES], fullConfig);
    std::unordered_set<std::string> supportedLayers;
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
                _supportedDevices.insert(value.deviceName);
            }
        } catch (...) {
            std::cout << value.deviceName << " doesn't support queryNetwork\n";
        }
    }

    if (_supportedDevices.empty()) {
        IE_THROW() << "Please, check environment due to no supported devices can be used";
    }
    std::cout << "AUTO supported devices: ";
    std::copy(_supportedDevices.begin(), _supportedDevices.end(), std::ostream_iterator<std::string>(std::cout, " "));
    std::cout << std::endl;

    for (auto&& supportedLayer : supportedLayers) {
        queryResult.supportedLayersMap[supportedLayer] = GetName();
    }
    return queryResult;
}

}  // namespace AutoPlugin
