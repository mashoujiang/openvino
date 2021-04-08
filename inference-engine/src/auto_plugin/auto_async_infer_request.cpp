// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include <string>
#include <vector>
#include <memory>
#include <map>

#include "auto_async_infer_request.hpp"

namespace AutoPlugin {
    using namespace InferenceEngine;

AutoAsyncInferRequest::AutoAsyncInferRequest(
    const AutoInferRequest::Ptr&         inferRequest,
    const bool                                  needPerfCounters,
    const AutoExecutableNetwork::Ptr&    autoExecutableNetwork,
    const ITaskExecutor::Ptr&                   callbackExecutor) :
    AsyncInferRequestThreadSafeDefault(inferRequest, nullptr, callbackExecutor),
    _autoExecutableNetwork{autoExecutableNetwork},
    _inferRequest{inferRequest},
    _needPerfCounters{needPerfCounters} {
    // this executor starts the inference while  the task (checking the result) is passed to the next stage
    struct ThisRequestExecutor : public ITaskExecutor {
        explicit ThisRequestExecutor(AutoAsyncInferRequest* _this_) : _this{_this_} {}
        void run(Task task) override {
            auto workerInferRequest = _this->_workerInferRequest;
            workerInferRequest->_task = std::move(task);
            workerInferRequest->_inferRequest.StartAsync();
        };
        AutoAsyncInferRequest* _this = nullptr;
    };
    _pipeline = {
        // if the request is coming with device-specific remote blobs make sure it is scheduled to the specific device only:
        { /*TaskExecutor*/ std::make_shared<ImmediateExecutor>(), /*task*/ [this] {
               // by default, no preferred device:
               _autoExecutableNetwork->_thisPreferredDeviceName = "";
               // if any input is remote (e.g. was set with SetBlob), let' use the corresponding device
               for (const auto &it : _autoExecutableNetwork->GetInputsInfo()) {
                   auto b = _inferRequest->GetBlob(it.first);
                   auto r = b->as<RemoteBlob>();
                   if (r) {
                       const auto name = r->getDeviceName();
                       const auto res = std::find_if(
                               _autoExecutableNetwork->_devicePrioritiesInitial.cbegin(),
                               _autoExecutableNetwork->_devicePrioritiesInitial.cend(),
                               [&name](const AutoPlugin::DeviceInformation& d){ return d.deviceName == name; });
                       if (_autoExecutableNetwork->_devicePrioritiesInitial.cend() == res) {
                           IE_THROW() << "None of the devices (for which current AUTO-device configuration was "
                                                 "initialized) supports a remote blob created on the device named " << name;

                       } else {
                            // it is ok to take the c_str() here (as pointed in the auto_exec_network.hpp we need to use const char*)
                            // as the original strings are from the "persistent" vector (with the right lifetime)
                           _autoExecutableNetwork->_thisPreferredDeviceName = res->deviceName.c_str();
                           break;
                       }
                   }
               }
        }},
        // as the scheduling algo may select any device, this stage accepts the scheduling decision (actual workerRequest)
        // then sets the device-agnostic blobs to the actual (device-specific) request
        {
         /*TaskExecutor*/ _autoExecutableNetwork, /*task*/ [this] {
               _workerInferRequest = AutoExecutableNetwork::_thisWorkerInferRequest;
               _inferRequest->SetBlobsToAnotherRequest(_workerInferRequest->_inferRequest);
        }},
        // final task in the pipeline:
        { /*TaskExecutor*/std::make_shared<ThisRequestExecutor>(this), /*task*/ [this] {
              auto status = _workerInferRequest->_status;
              if (InferenceEngine::StatusCode::OK != status) {
                  if (nullptr != InferenceEngine::CurrentException())
                      std::rethrow_exception(InferenceEngine::CurrentException());
                  else
                      IE_EXCEPTION_SWITCH(status, ExceptionType,
                        InferenceEngine::details::ThrowNow<ExceptionType>{}
                            <<= std::stringstream{} << IE_LOCATION
                            <<  InferenceEngine::details::ExceptionTraits<ExceptionType>::string());
              }
              if (_needPerfCounters)
                  _perfMap = _workerInferRequest->_inferRequest.GetPerformanceCounts();
        }}
    };
}

void AutoAsyncInferRequest::Infer_ThreadUnsafe() {
    InferUsingAsync();
}

std::map<std::string, InferenceEngineProfileInfo> AutoAsyncInferRequest::GetPerformanceCounts() const {
    CheckState();
    return std::move(_perfMap);
}

AutoAsyncInferRequest::~AutoAsyncInferRequest() {
    StopAndWait();
}

}  // namespace AutoPlugin
