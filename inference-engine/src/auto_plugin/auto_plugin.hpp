// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <map>
#include <vector>
#include <string>
#include <unordered_set>

#include <cpp_interfaces/impl/ie_plugin_internal.hpp>
#include <cpp_interfaces/interface/ie_internal_plugin_config.hpp>
#include "auto_exec_network.hpp"

namespace AutoPlugin {

class AutoInferencePlugin : public InferenceEngine::InferencePluginInternal {
public:
    AutoInferencePlugin();
    ~AutoInferencePlugin() = default;

    InferenceEngine::ExecutableNetworkInternal::Ptr LoadExeNetworkImpl(const InferenceEngine::CNNNetwork&        network,
                                                                       const std::map<std::string, std::string>& config) override;

    void SetConfig(const std::map<std::string, std::string>& config) override;
    InferenceEngine::Parameter GetConfig(const std::string& name, const std::map<std::string, InferenceEngine::Parameter> & options) const override;
    InferenceEngine::QueryNetworkResult QueryNetwork(const InferenceEngine::CNNNetwork&        network,
                                                     const std::map<std::string, std::string>& config) const override;
    InferenceEngine::Parameter GetMetric(const std::string& name,
                                         const std::map<std::string, InferenceEngine::Parameter>& options) const override;

    std::vector<AutoPlugin::DeviceInformation> ParseMetaDevices(const std::string & devicesRequestsCfg,
                                                                const std::map<std::string, std::string> & config) const;

private:
    static std::string GetPriorityDevices();
    std::vector<std::string> GetOptimizationCapabilities() const;
    std::vector<DeviceInformation>::const_iterator  SelectDevicePolicy(const std::vector<AutoPlugin::DeviceInformation>& metaDevices) const;

protected:
    std::map<std::string, std::string> GetSupportedConfig(const std::map<std::string, std::string>& config,
                                                          const AutoPlugin::DeviceName & deviceName) const;

private:
    mutable std::unordered_set<std::string> _supportedDevices;
};

}  // namespace AutoPlugin
