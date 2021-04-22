// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include <mutex>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <map>
#include <unordered_map>


#include "ie_metric_helpers.hpp"
#include <cpp_interfaces/base/ie_infer_async_request_base.hpp>
#include <auto_plugin/auto_config.hpp>
#include <ie_plugin_config.hpp>
#include "auto_exec_network.hpp"
#include "auto_async_infer_request.hpp"
#include "auto_plugin.hpp"

// ------------------------------AutoExecutableNetwork----------------------------
namespace AutoPlugin {
    using namespace InferenceEngine;

thread_local AutoExecutableNetwork::WorkerInferRequest* AutoExecutableNetwork::_thisWorkerInferRequest = nullptr;

struct IdleGuard {
    explicit IdleGuard(AutoExecutableNetwork::WorkerInferRequest* workerInferRequestPtr,
                       AutoExecutableNetwork::NotBusyWorkerRequests& notBusyWorkerRequests) :
        _workerInferRequestPtr{workerInferRequestPtr},
        _notBusyWorkerRequests{&notBusyWorkerRequests} {
    }
    ~IdleGuard() {
        if (nullptr != _notBusyWorkerRequests) {
            _notBusyWorkerRequests->try_push(_workerInferRequestPtr);
        }
    }
    AutoExecutableNetwork::NotBusyWorkerRequests* Release() {
        auto notBusyWorkerRequests = _notBusyWorkerRequests;
        _notBusyWorkerRequests = nullptr;
        return notBusyWorkerRequests;
    }
    AutoExecutableNetwork::WorkerInferRequest*     _workerInferRequestPtr = nullptr;
    AutoExecutableNetwork::NotBusyWorkerRequests*  _notBusyWorkerRequests = nullptr;
};

AutoExecutableNetwork::AutoExecutableNetwork(const InferenceEngine::ExecutableNetwork&                                          network,
                                                           const DeviceInformation&                                             deviceInfo,
                                                           const std::unordered_map<std::string, InferenceEngine::Parameter>&   config,
                                                           const bool                                                           needPerfCounters) :
    InferenceEngine::ExecutableNetworkThreadSafeDefault(nullptr, std::make_shared<InferenceEngine::ImmediateExecutor>()),
    _deviceInfo{deviceInfo},
    _deviceInfoInitial{deviceInfo},
    _network{network},
    _config{config},
    _needPerfCounters{needPerfCounters} {
    _taskExecutor.reset();

    const auto& deviceName = _deviceInfo.deviceName;
    unsigned int optimalNum = 0;
    try {
        optimalNum = _network.GetMetric(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)).as<unsigned int>();
    } catch (const InferenceEngine::Exception &iie) {
    }
    const auto numRequests = _deviceInfo.numRequests == -1 ? (optimalNum == 0 ? 1: optimalNum) : _deviceInfo.numRequests;
    _workerRequests.resize(numRequests);
    auto* idleWorkerRequestsPtr = &(_idleWorkerRequests);
    _idleWorkerRequests.set_capacity(numRequests);
    for (auto&& workerRequest : _workerRequests) {
        workerRequest._inferRequest = _network.CreateInferRequest();
        auto* workerRequestPtr = &workerRequest;
        IE_ASSERT(_idleWorkerRequests.try_push(workerRequestPtr) == true);
        workerRequest._inferRequest.SetCompletionCallback<std::function<void(InferRequest, StatusCode)>>(
            [workerRequestPtr, this, deviceName, idleWorkerRequestsPtr] (InferRequest , StatusCode status) mutable {
                IdleGuard idleGuard{workerRequestPtr, *idleWorkerRequestsPtr};
                workerRequestPtr->_status = status;
                {
                    auto capturedTask = std::move(workerRequestPtr->_task);
                    capturedTask();
                }
                // try to return the request to the idle list (fails if the overall object destruction has began)
                if (idleGuard.Release()->try_push(workerRequestPtr)) {
                    // let's try to pop a task, as we know there is at least one idle request, schedule if succeeded
                    // if no device-agnostic tasks, let's try pop the device specific task, schedule if succeeded
                    Task t;
                    if (_inferPipelineTasks.try_pop(t))
                        ScheduleToWorkerInferRequest(std::move(t));
                }
            });
    }
}


void AutoExecutableNetwork::ScheduleToWorkerInferRequest(Task inferPipelineTask) {
    WorkerInferRequest *workerRequestPtr = nullptr;
    NotBusyWorkerRequests &idleWorkerRequests = _idleWorkerRequests;
    if (idleWorkerRequests.try_pop(workerRequestPtr)) {
        IdleGuard idleGuard{workerRequestPtr, idleWorkerRequests};
        _thisWorkerInferRequest = workerRequestPtr;
        {
            auto capturedTask = std::move(inferPipelineTask);
            capturedTask();
        }
        idleGuard.Release();
        return;
    }

    // no vacant requests this time, storing the task to the respective queue
    _inferPipelineTasks.push(std::move(inferPipelineTask));
}

void AutoExecutableNetwork::run(Task inferPipelineTask) {
    ScheduleToWorkerInferRequest(std::move(inferPipelineTask));
}

AutoExecutableNetwork::~AutoExecutableNetwork() {
    // stop accepting any idle requests back (for re-scheduling)
    _idleWorkerRequests.set_capacity(0);
    _workerRequests.clear();
}

RemoteContext::Ptr AutoExecutableNetwork::GetContext() const {
    try {
        return _network.GetContext();
    } catch (const NotImplemented&) {
        IE_THROW(NotImplemented) << "None of the devices in the AUTO has an associated remote context."
                                 << " Current list of devices allowed via the DEVICE_PRIORITIES config: " << _deviceInfo.deviceName;
    }
}

InferenceEngine::InferRequestInternal::Ptr AutoExecutableNetwork::CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                                                                                         InferenceEngine::OutputsDataMap networkOutputs) {
    auto num = _numRequestsCreated++;
    InferenceEngine::InferRequest request_to_share_blobs_with;

    auto& dev_requests = _workerRequests;
    if (num < dev_requests.size()) {
        request_to_share_blobs_with = dev_requests.at(num)._inferRequest;
    }

    return std::make_shared<AutoInferRequest>(networkInputs, networkOutputs, request_to_share_blobs_with);
}

IInferRequest::Ptr AutoExecutableNetwork::CreateInferRequest() {
    IInferRequest::Ptr asyncRequest;
    auto syncRequestImpl = CreateInferRequestImpl(_networkInputs, _networkOutputs);
    syncRequestImpl->setPointerToExecutableNetworkInternal(shared_from_this());
    auto asyncTreadSafeImpl = std::make_shared<AutoAsyncInferRequest>(std::static_pointer_cast<AutoInferRequest>(syncRequestImpl),
                                                                             _needPerfCounters,
                                                                             std::static_pointer_cast<AutoExecutableNetwork>(shared_from_this()),
                                                                             _callbackExecutor);
    asyncRequest.reset(new InferRequestBase(asyncTreadSafeImpl));
    asyncTreadSafeImpl->SetPointerToPublicInterface(asyncRequest);
    return asyncRequest;
}

void AutoExecutableNetwork::SetConfig(const std::map<std::string, InferenceEngine::Parameter> &config) {
    IE_THROW(NotImplemented) << "Auto plugin doesn't implement SetConfig";
}

InferenceEngine::Parameter AutoExecutableNetwork::GetConfig(const std::string &name) const {
    auto it = _config.find(name);
    if (it != _config.end()) {
        return it->second;
    } else {
        IE_THROW(NotFound) << name <<" not found in the ExecutableNetwork config";
    }
}

InferenceEngine::Parameter AutoExecutableNetwork::GetMetric(const std::string &name) const {
    if (name == METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)) {
        unsigned int res = 0u;
            try {
              res = _network.GetMetric(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)).as<unsigned int>();
            } catch (const InferenceEngine::Exception &iie) {
                  IE_THROW()
                        << "Every device used with the Auto-Device should "
                        << "support OPTIMAL_NUMBER_OF_INFER_REQUESTS ExecutableNetwork metric. "
                        << "Failed to query the metric for the " << _deviceInfo.deviceName << " with error:" << iie.what();
           }

        IE_SET_METRIC_RETURN(OPTIMAL_NUMBER_OF_INFER_REQUESTS, res);
    } else if (name == METRIC_KEY(NETWORK_NAME)) {
        IE_SET_METRIC_RETURN(NETWORK_NAME, _network.GetMetric(
            METRIC_KEY(NETWORK_NAME)).as<std::string>());
    } else if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, {
            METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS),
            METRIC_KEY(SUPPORTED_METRICS),
            METRIC_KEY(NETWORK_NAME),
            METRIC_KEY(SUPPORTED_CONFIG_KEYS)
        });
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        std::vector<std::string> configKeys = { AutoConfigParams::KEY_AUTO_DEVICE_PRIORITIES };
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, configKeys);
    } else {
        IE_THROW() << "Unsupported Network metric: " << name;
    }
}

}  // namespace AutoPlugin
