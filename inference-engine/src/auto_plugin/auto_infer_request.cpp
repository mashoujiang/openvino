// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "auto_infer_request.hpp"
#include <ie_input_info.hpp>
#include <ie_icnn_network.hpp>
#include <cpp_interfaces/interface/ie_iinfer_request_internal.hpp>
#include <blob_factory.hpp>

namespace AutoPlugin {
    using namespace InferenceEngine;
// ------------------------------AutoInferRequest----------------------------
AutoInferRequest::AutoInferRequest(const InputsDataMap&   networkInputs,
                                                 const OutputsDataMap&  networkOutputs,
                                                 InferRequest request_to_share_blobs_with)
        : IInferRequestInternal(networkInputs, networkOutputs) {
    if (request_to_share_blobs_with) {
        // borrow device-friendly blobs from the request
        for (const auto &it : _networkInputs)
            _inputs[it.first] = request_to_share_blobs_with.GetBlob(it.first);
        for (const auto &it : _networkOutputs)
            _outputs[it.first] = request_to_share_blobs_with.GetBlob(it.first);
        return;
    }
    // Allocate all input blobs
    for (const auto &it : networkInputs) {
        Layout l = it.second->getLayout();
        Precision p = it.second->getPrecision();
        SizeVector dims = it.second->getTensorDesc().getDims();

        TensorDesc desc = TensorDesc(p, dims, l);
        _inputs[it.first] = make_blob_with_precision(desc);
        _inputs[it.first]->allocate();
    }
    // Allocate all output blobs
    for (const auto &it : networkOutputs) {
        Layout l = it.second->getLayout();
        Precision p = it.second->getPrecision();
        SizeVector dims = it.second->getTensorDesc().getDims();

        TensorDesc desc = TensorDesc(p, dims, l);
        _outputs[it.first] = make_blob_with_precision(desc);
        _outputs[it.first]->allocate();
    }
}

void AutoInferRequest::SetBlobsToAnotherRequest(InferRequest& req) {
    for (const auto &it : _networkInputs) {
        auto &name = it.first;
        // this request is already in BUSY state, so using the internal functions safely
        auto blob = GetBlob(name);
        if (req.GetBlob(name) != blob)
            req.SetBlob(name, blob);
    }
    for (const auto &it : _networkOutputs) {
        auto &name = it.first;
        // this request is already in BUSY state, so using the internal functions safely
        auto blob = GetBlob(name);
        if (req.GetBlob(name) != blob)
            req.SetBlob(name, blob);
    }
}

std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> AutoInferRequest::GetPerformanceCounts() const {
    IE_THROW(NotImplemented);
}

void AutoInferRequest::InferImpl() {
    IE_THROW(NotImplemented);
}

}  // namespace AutoPlugin
