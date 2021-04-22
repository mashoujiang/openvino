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
#include "auto_schedule_policy.hpp"

namespace AutoPlugin {

class AutoInferencePlugin : public InferenceEngine::InferencePluginInternal {
public:
    AutoInferencePlugin();
    ~AutoInferencePlugin() = default;

    InferenceEngine::ExecutableNetworkInternal::Ptr LoadExeNetworkImpl(const InferenceEngine::CNNNetwork&        network,
                                                                       const std::map<std::string, std::string>& config) override;
    InferenceEngine::QueryNetworkResult QueryNetwork(const InferenceEngine::CNNNetwork&        network,
                                                     const std::map<std::string, std::string>& config) const override;

    void SetConfig(const std::map<std::string, std::string>& config) override;
    InferenceEngine::Parameter GetConfig(const std::string& name, const std::map<std::string, InferenceEngine::Parameter> & options) const override;

    InferenceEngine::Parameter GetMetric(const std::string& name,
                                         const std::map<std::string, InferenceEngine::Parameter>& options) const override;

private:
    static std::string GetPriorityDevices();
    static SchedulePolicyType ParseScheduleType(const std::string & scheduleType);

    void RegisterPolicy(SchedulePolicyType type);
    std::vector<std::string> GetOptimizationCapabilities() const;
    std::vector<AutoPlugin::DeviceInformation> ParseMetaDevices(const std::string & devicesRequestsCfg,
                                                                const std::map<std::string, std::string> & config) const;

protected:
    std::map<std::string, std::string> GetSupportedConfig(const std::map<std::string, std::string>& config,
                                                          const AutoPlugin::DeviceName & deviceName) const;

private:
    std::unordered_map<SchedulePolicyType, AutoSchedulePolicy::Ptr> _policies;
};

}  // namespace AutoPlugin
