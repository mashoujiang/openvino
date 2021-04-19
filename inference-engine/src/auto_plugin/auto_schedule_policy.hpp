// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Auto plugin schedule policy base definition
 */

#pragma once

#include "auto_exec_network.hpp"
#include <cpp/ie_cnn_network.h>

namespace AutoPlugin {

using VecDevice = std::vector<DeviceInformation>;
using VecDeviceCiter = std::vector<DeviceInformation>::const_iterator;

enum class SchedulePolicyType {
  STATIC        = 0,
  THROUGH_PUT   = 1,
  LATENCY       = 2
};

class AutoSchedulePolicy{
public:
    using Ptr = std::unique_ptr<AutoSchedulePolicy>;
    explicit AutoSchedulePolicy(SchedulePolicyType type);
    ~AutoSchedulePolicy();

    VecDeviceCiter SelectDevice(const InferenceEngine::CNNNetwork &network, const VecDevice& metaDevices) const;

    class Priv;

private:
    static std::string StrPolicy(SchedulePolicyType type);

private:
    std::unique_ptr<Priv> _priv;
};
} // namespace AutoPlugin