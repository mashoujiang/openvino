// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header that defines advanced related properties for Auto plugin.
 * These properties should be used in SetConfig() and LoadNetwork() methods
 *
 * @file auto_config.hpp
 */

#pragma once

#include "ie_plugin_config.hpp"

namespace InferenceEngine {

/**
 * @brief Auto plugin configuration
 */
namespace AutoConfigParams {

/**
 * @def AUTO_CONFIG_KEY(name)
 * @brief A macro which provides a AUTO-mangled name for configuration key with name `name`
 */
#define AUTO_CONFIG_KEY(name) InferenceEngine::AutoConfigParams::_CONFIG_KEY(AUTO_##name)

#define DECLARE_AUTO_CONFIG_KEY(name) DECLARE_CONFIG_KEY(AUTO_##name)
#define DECLARE_AUTO_CONFIG_VALUE(name) DECLARE_CONFIG_VALUE(AUTO_##name)

/**
 * @brief Device Priorities config option, with comma-separated devices listed in the desired priority
 */
DECLARE_AUTO_CONFIG_KEY(DEVICE_PRIORITIES);

}  // namespace AutoConfigParams
}  // namespace InferenceEngine
