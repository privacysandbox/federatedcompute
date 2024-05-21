// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.fcp.plan;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableMap;
import com.google.protobuf.ByteString;
import org.tensorflow.framework.TensorProto;

/** Objects that implement this interface execute the ServerPhaseV2 portion of a Plan object. */
public interface PhaseSessionV2 {
    /**
     * The combined updated client checkpoint and intermediate state produced by executing the server_prepare
     * logic in a ServerPhaseV2 message.
     */
    @AutoValue
    public abstract static class IntermediateResult {
        public abstract ByteString clientCheckpoint();

        public abstract ByteString interemdiateState();

        public static IntermediateResult create(
                ByteString clientCheckpoint, ByteString interemdiateState) {
            return new AutoValue_PhaseSessionV2_IntermediateResult(clientCheckpoint, interemdiateState);
        }
    }

    /**
     * Run the server_prepare logic, which takes the current server state and generates the client
     * checkpoint and intermediate state.
     */
    public IntermediateResult getClientCheckpoint(ByteString serverState);

    /**
     * The combined updated server state and output metrics produced by executing the server_result
     * logic in a ServerPhaseV2 message.
     */
    @AutoValue
    public abstract static class Result {
        public abstract ByteString updatedServerState();

        public abstract ImmutableMap<String, TensorProto> metrics();

        public static Result create(
                ByteString updatedServerState, ImmutableMap<String, TensorProto> metrics) {
            return new AutoValue_PhaseSessionV2_Result(updatedServerState, metrics);
        }
    }

    /**
     * Run the server_result logic, which takes the aggregated client result and the intermediate
     * state as input and generates the updated server state and other outputs.
     */
    public Result getResult(ByteString aggregateResult, ByteString intermediateState);
}
