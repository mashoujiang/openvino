// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "auto_schedule_policy.hpp"

namespace AutoPlugin {

class AutoSchedulePolicy::Priv{
public:
    virtual ~Priv() = default;
    virtual VecDeviceCiter SelectDevicePolicy(const VecDevice& metaDevices) const = 0;
};

class AutoStaticPolicy: public AutoSchedulePolicy::Priv{
public:
    VecDeviceCiter SelectDevicePolicy(const VecDevice& metaDevices) const override;
};

VecDeviceCiter AutoStaticPolicy::SelectDevicePolicy(const VecDevice& metaDevices) const {
    return metaDevices.begin();
}

AutoSchedulePolicy::AutoSchedulePolicy(SchedulePolicyType type) {
    switch (type) {
        case SchedulePolicyType::STATIC: {
            _priv.reset(new AutoStaticPolicy());
            break;
        }
        case SchedulePolicyType::THROUGH_PUT:
        case SchedulePolicyType::LATENCY:
        default: {
            IE_THROW(NotImplemented)
                << "Does not implement schedule type " << StrPolicy(type);
        }
    }
}

AutoSchedulePolicy::~AutoSchedulePolicy() = default;

VecDeviceCiter AutoSchedulePolicy::SelectDevicePolicy(const VecDevice& metaDevices) const {
    return _priv->SelectDevicePolicy(metaDevices);
}

std::string AutoSchedulePolicy::StrPolicy(SchedulePolicyType type) {
    switch (type) {
    case SchedulePolicyType::STATIC:
        return "STATIC";
    case SchedulePolicyType::THROUGH_PUT:
        return "THROUGH_PUT";
    case SchedulePolicyType::LATENCY:
        return "LATENCY";
    default:
        return "UNSUPPORT";
    }
}
} // namespace AutoPlugin