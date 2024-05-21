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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.fcp.tensorflow.AppFiles;
import com.google.fcp.tensorflow.TensorflowSession;
import com.google.internal.federated.plan.Plan;
import com.google.internal.federated.plan.ServerPhaseV2;
import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;
import org.tensorflow.framework.TensorProto;
import tensorflow.Struct.TensorSpecProto;

import java.io.File;
import java.io.FileInputStream;
import java.util.HashSet;
import java.util.Map;
import java.util.Optional;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;
import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.ImmutableMap.toImmutableMap;
import static com.google.common.collect.ImmutableSet.toImmutableSet;

/**
 * Executes the ServerPhaseV2 portion of an execution plan (a Plan object).
 *
 * <p>Separate TensorFlow sessions are used to run the server_prepare and server_result portions of
 * the ServerPhaseV2 message. Any state that is needed to connect these two TensorFlow sessions is
 * stored in memory in the intermediate state file.
 */
public class TensorflowPhaseSessionV2 implements PhaseSessionV2 {
    private static final AppFiles fileCache = new AppFiles("/tmp");

    private final ServerPhaseV2 phase;

    // The TensorFlow graph for running the server_prepare logic.
    private Optional<byte[]> serverPrepareGraph = Optional.empty();

    // The TensorFlow graph for running the server_result logic.
    private Optional<byte[]> serverResultGraph = Optional.empty();

    public TensorflowPhaseSessionV2(ByteString planBytes) {
        checkNotNull(planBytes);
        checkArgument(!planBytes.isEmpty(), "PlanBytes cannot be empty");

        Plan plan;
        try {
            plan = Plan.parseFrom(planBytes);
        } catch (InvalidProtocolBufferException e) {
            throw new IllegalArgumentException("Failed to decode TensorFlow execution plan.", e);
        }

        if (plan.getPhaseList().isEmpty()) {
            throw new IllegalArgumentException("Found plan with no phases.");
        }

        if (!plan.getPhase(0).hasServerPhaseV2()) {
            throw new IllegalArgumentException("Found plan without ServerPhaseV2 message.");
        }

        this.phase = plan.getPhase(0).getServerPhaseV2();

        if (plan.hasServerGraphPrepareBytes() != phase.hasTensorflowSpecPrepare()
                || plan.hasServerGraphPrepareBytes() != phase.hasPrepareRouter()) {
            throw new IllegalArgumentException("Found plan with inconsistent server_prepare fields.");
        }

        if (plan.hasServerGraphResultBytes() != phase.hasTensorflowSpecResult()
                || plan.hasServerGraphResultBytes() != phase.hasResultRouter()) {
            throw new IllegalArgumentException("Found plan with inconsistent server_result fields.");
        }

        if (plan.hasServerGraphPrepareBytes()) {
            this.serverPrepareGraph =
                    Optional.of(plan.getServerGraphPrepareBytes().getValue().toByteArray());
        }

        if (plan.hasServerGraphResultBytes()) {
            this.serverResultGraph =
                    Optional.of(plan.getServerGraphResultBytes().getValue().toByteArray());
        }
    }

    // Run the server_prepare logic, which takes the current server state and generates the client
    // checkpoint and intermediate state.
    @Override
    public IntermediateResult getClientCheckpoint(ByteString serverState) {
        // Return an empty checkpoint if there is no server_prepare logic.
        if (serverPrepareGraph.isEmpty()) {
            return IntermediateResult.create(ByteString.EMPTY, ByteString.EMPTY);
        }
        File serverStateInputFile = null;
        File clientCheckpointOutputFile = null;
        File intermediateStateOutputFile = null;
        try (TensorflowSession session =
                     TensorflowSession.createFromByteArray(fileCache, serverPrepareGraph.get())) {
            // Assemble the inputs for running the server_prepare logic.
            ImmutableMap.Builder<String, TensorProto> prepareInputsBuilder = ImmutableMap.builder();

            // Create a temporary file from which the current server state can be read.
            serverStateInputFile = session.writeTempCheckpointFile(serverState);
            prepareInputsBuilder.put(
                    phase.getPrepareRouter().getPrepareServerStateInputFilepathTensorName(),
                    TensorflowSession.stringTensor(serverStateInputFile.getPath()));

            // Create a temporary file to which the client checkpoint can be written.
            clientCheckpointOutputFile = fileCache.createTempFile("client_checkpoint", ".ckp");
            prepareInputsBuilder.put(
                    phase.getPrepareRouter().getPrepareOutputFilepathTensorName(),
                    TensorflowSession.stringTensor(clientCheckpointOutputFile.getPath()));

            // Create a temporary file to which the intermediate state can be written.
            intermediateStateOutputFile = fileCache.createTempFile("intermediate_state", ".ckp");
            prepareInputsBuilder.put(
                    phase.getPrepareRouter().getPrepareIntermediateStateOutputFilepathTensorName(),
                    TensorflowSession.stringTensor(intermediateStateOutputFile.getPath()));

            ImmutableMap<String, TensorProto> prepareInputs = prepareInputsBuilder.buildOrThrow();

            // The names of the input tensors specified by the ServerPrepareIORouter should match the
            // server_prepare TensorflowSpec input names.
            ImmutableSet<String> expectedTensorInputNames =
                    phase.getTensorflowSpecPrepare().getInputTensorSpecsList().stream()
                            .map(TensorSpecProto::getName)
                            .collect(toImmutableSet());
            checkState(
                    expectedTensorInputNames.equals(prepareInputs.keySet()),
                    "Prepared input tensor names do not match tensorflow_spec_prepare input names.");

            // The server_prepare logic runs a TensorFlow session using target nodes and no outputs.
            checkState(
                    !phase.getTensorflowSpecPrepare().getTargetNodeNamesList().isEmpty(),
                    "Found tensorflow_spec_prepare with empty target nodes.");
            checkState(
                    phase.getTensorflowSpecPrepare().getOutputTensorSpecsList().isEmpty(),
                    "Found tensorflow_spec_prepare with unexpected outputs.");

            // Run the server_prepare logic.
            session.run(
                    prepareInputs,
                    /* outputNames= */ null,
                    phase.getTensorflowSpecPrepare().getTargetNodeNamesList());

            ByteString intermediateState = null;
            // Read the intermediate state into memory.
            try (FileInputStream intermediateStateOutputFileInputStream =
                         new FileInputStream(intermediateStateOutputFile)) {
                intermediateState = ByteString.readFrom(intermediateStateOutputFileInputStream);
            }

            ByteString clientCheckpoint = null;
            // Read the client checkpoint output file.
            try (FileInputStream clientCheckpointOutputFileInputStream =
                         new FileInputStream(clientCheckpointOutputFile)) {
                clientCheckpoint = ByteString.readFrom(clientCheckpointOutputFileInputStream);
            }
            return IntermediateResult.create(clientCheckpoint, intermediateState);
        } catch (Exception e) {
            throw new IllegalStateException(e);
        } finally {
            // Allow the field storing the server_prepare TF graph to be garbage collected.
            serverPrepareGraph = null;

            // Always delete the temp files before exiting this method.
            if (serverStateInputFile != null) {
                serverStateInputFile.delete();
            }
            if (clientCheckpointOutputFile != null) {
                clientCheckpointOutputFile.delete();
            }
            if (intermediateStateOutputFile != null) {
                intermediateStateOutputFile.delete();
            }
        }
    }

    // Run the server_result logic, which takes the aggregated client result and the intermediate
    // state as input and generates the updated server state and other outputs.
    @Override
    public Result getResult(ByteString aggregateResult, ByteString intermediateState) {
        // Return the aggregated client result and no metrics if there is no server_result logic.
        if (serverResultGraph.isEmpty()) {
            return Result.create(aggregateResult, ImmutableMap.of());
        }
        File intermediateStateInputFile = null;
        File aggregateResultInputFile = null;
        File serverStateOutputFile = null;
        try (TensorflowSession session =
                     TensorflowSession.createFromByteArray(fileCache, serverResultGraph.get())) {
            // Assemble the inputs for running the server_result logic.
            ImmutableMap.Builder<String, TensorProto> resultInputsBuilder = ImmutableMap.builder();

            // Create a temporary file from which the intermediate state can be read.
            intermediateStateInputFile = session.writeTempCheckpointFile(intermediateState);
            resultInputsBuilder.put(
                    phase.getResultRouter().getResultIntermediateStateInputFilepathTensorName(),
                    TensorflowSession.stringTensor(intermediateStateInputFile.getPath()));

            // Create a temporary file from which the aggregated client results can be read.
            aggregateResultInputFile = session.writeTempCheckpointFile(aggregateResult);
            resultInputsBuilder.put(
                    phase.getResultRouter().getResultAggregateResultInputFilepathTensorName(),
                    TensorflowSession.stringTensor(aggregateResultInputFile.getPath()));

            // Create a temporary file to which the updated server state can be written.
            serverStateOutputFile = fileCache.createTempFile("server_state", ".ckp");
            resultInputsBuilder.put(
                    phase.getResultRouter().getResultServerStateOutputFilepathTensorName(),
                    TensorflowSession.stringTensor(serverStateOutputFile.getPath()));

            ImmutableMap<String, TensorProto> resultInputs = resultInputsBuilder.buildOrThrow();

            // The names of the input tensors specified by the ServerResultIORouter should match the
            // server_result TensorflowSpec input names.
            ImmutableSet<String> expectedTensorInputNames =
                    phase.getTensorflowSpecResult().getInputTensorSpecsList().stream()
                            .map(TensorSpecProto::getName)
                            .collect(toImmutableSet());
            checkState(
                    expectedTensorInputNames.equals(resultInputs.keySet()),
                    "Prepared input tensor names do not match tensorflow_spec_result input names.");

            // The server_result logic runs a TensorFlow session by specifying target nodes and
            // potentially also output tensors.
            checkState(
                    !phase.getTensorflowSpecResult().getTargetNodeNamesList().isEmpty(),
                    "Found tensorflow_spec_result with empty target nodes.");

            // Run the server_result logic.
            ImmutableList<String> outputNames =
                    phase.getTensorflowSpecResult().getOutputTensorSpecsList().stream()
                            .map(TensorSpecProto::getName)
                            .collect(toImmutableList());
            Map<String, TensorProto> outputMetrics =
                    session.run(
                            resultInputs, outputNames, phase.getTensorflowSpecResult().getTargetNodeNamesList());

            // Check that the output contains the expected tensor names.
            checkState(
                    outputMetrics.keySet().equals(new HashSet<String>(outputNames)),
                    "Output tensors do not match expected output tensor names.");

            // Read the update server state checkpoint output file and combine it with the output metrics
            // to produce the result. When creating the output metrics map, remove the colon suffix from
            // the tensor names.
            try (FileInputStream serverStateOutputFileInputStream =
                         new FileInputStream(serverStateOutputFile)) {
                return Result.create(
                        ByteString.readFrom(serverStateOutputFileInputStream),
                        outputMetrics.entrySet().stream()
                                .collect(
                                        toImmutableMap(
                                                entry -> getBareTensorName(entry.getKey()), Map.Entry::getValue)));
            }
        } catch (Exception e) {
            throw new IllegalStateException(e);
        } finally {
            // Allow the field storing the server_result TF graph to be garbage collected.
            serverResultGraph = null;

            // Always delete the temp files.
            if (intermediateStateInputFile != null) {
                intermediateStateInputFile.delete();
            }
            if (aggregateResultInputFile != null) {
                aggregateResultInputFile.delete();
            }
            if (serverStateOutputFile != null) {
                serverStateOutputFile.delete();
            }
        }
    }

    // Removes the colon suffix from a tensor name, if it exists.
    private String getBareTensorName(String tensorName) {
        int colon = tensorName.indexOf(':');
        if (colon >= 0) {
            return tensorName.substring(0, colon);
        }
        return tensorName;
    }
}
