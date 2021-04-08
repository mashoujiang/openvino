// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <map>
#include <vector>
#include <utility>
#include <memory>
#include <string>

#include <cpp_interfaces/impl/ie_infer_async_request_thread_safe_default.hpp>
#include "auto_infer_request.hpp"
#include "auto_exec_network.hpp"

namespace AutoPlugin {

class AutoAsyncInferRequest : public InferenceEngine::AsyncInferRequestThreadSafeDefault {
public:
    using Ptr = std::shared_ptr<AutoAsyncInferRequest>;

    explicit AutoAsyncInferRequest(const AutoInferRequest::Ptr&           inferRequest,
                                          const bool                                    needPerfCounters,
                                          const AutoExecutableNetwork::Ptr&      autoExecutableNetwork,
                                          const InferenceEngine::ITaskExecutor::Ptr&    callbackExecutor);
    void Infer_ThreadUnsafe() override;
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> GetPerformanceCounts() const override;
    ~AutoAsyncInferRequest() override;

protected:
    AutoExecutableNetwork::Ptr                                   _autoExecutableNetwork;
    AutoInferRequest::Ptr                                        _inferRequest;
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>  _perfMap;
    bool                                                                _needPerfCounters = false;
    AutoExecutableNetwork::WorkerInferRequest*                   _workerInferRequest = nullptr;
};

}  // namespace AutoPlugin
