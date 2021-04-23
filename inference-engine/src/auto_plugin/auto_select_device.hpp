// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Auto plugin select device base definition
 */

#pragma once

#include "auto_exec_network.hpp"
#include <cpp/ie_cnn_network.h>

namespace AutoPlugin {

using VecDevice = std::vector<DeviceInformation>;
using VecDeviceCiter = std::vector<DeviceInformation>::const_iterator;

enum class SelectDevicePolicy {
    STATIC        = 0
};

class AutoSelectDevice{
public:
    using Ptr = std::unique_ptr<AutoSelectDevice>;
    explicit AutoSelectDevice(SelectDevicePolicy type);
    ~AutoSelectDevice();

    DeviceInformation SelectDevice(const InferenceEngine::CNNNetwork &network,
                                   const VecDevice& metaDevices,
                                   const std::vector<std::string>& optCap) const;

    class Priv;

private:
    static std::string StrPolicy(SelectDevicePolicy type);

private:
    std::unique_ptr<Priv> _priv;
};
} // namespace AutoPlugin