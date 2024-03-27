// Copyright 2023 Google LLC
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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.fcp.tensorflow.AppFiles;
import com.google.internal.federated.plan.CheckpointOp;
import com.google.internal.federated.plan.OutputMetric;
import com.google.internal.federated.plan.Plan;
import com.google.internal.federated.plan.Plan.Phase;
import com.google.internal.federated.plan.ServerPhase;
import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;
import java.util.Optional;

/**
 * Extracts and stores information from an execution plan (a Plan object).
 *
 * <p>This object is immutable after construction.
 */
public final class TensorflowPlanSession implements PlanSession {

  /**
   * Identifies the operation which saves / loads the server checkpoint.
   */
  final CheckpointOp serverSavepoint;

  /**
   * Stores phase information.
   */
  final Phase phase;

  /**
   * Stores plan information.
   */
  final ImmutableSet<OutputMetric> outputMetrics;

  /**
   * Plan version.
   */
  private final int version;

  /**
   * The serialized server execution graph. We retain this in serialized form because we pass this
   * across the JNI boundary. Once this is passed to the native implementation and parsed by C++
   * code, the local copy is no longer needed and gets nulled out to allow the memory to be garbage
   * collected.
   *
   * <p>NOTE: Although the Java type system cannot express this invariant, the contents of this
   * array should never be altered. This is conceptually a ByteString. We're storing it in a byte[]
   * so that we can avoid the conversion from ByteString to byte[] on the way through the JNI
   * layer.
   */
  private byte[] serverGraph;

  /**
   * This constructor extracts information from the execution plan that is required for the
   * operation of the service. It intentionally does *not* retain a reference to the Plan object.
   * Instead, it selects the subset of information that it needs, in the most efficient form that is
   * required for operation of the service.
   */
  public TensorflowPlanSession(ByteString planBytes) {
    Preconditions.checkNotNull(planBytes);
    Preconditions.checkArgument(!planBytes.isEmpty(), "PlanBytes cannot be empty");

    Plan plan;
    try {
      plan = Plan.parseFrom(planBytes);
    } catch (InvalidProtocolBufferException e) {
      throw new IllegalArgumentException("Failed to decode TensorFlow execution plan.", e);
    }

    if (plan.getPhaseList().isEmpty()) {
      throw new IllegalArgumentException("Expect plan to have at least one phase.");
    }

    // The server graph is stored in this class only for a very short time before it is passed
    // to a phase session object in createPhaseSession().
    if (plan.hasServerGraphBytes()) {
      this.serverGraph = plan.getServerGraphBytes().getValue().toByteArray();
    } else {
      this.serverGraph = new byte[0];
    }

    this.serverSavepoint = plan.getServerSavepoint();
    this.phase = plan.getPhase(0);
    this.version = plan.getVersion();

    if (plan.getOutputMetricsList().isEmpty()) {
      this.outputMetrics = ImmutableSet.of();
    } else {
      this.outputMetrics = ImmutableSet.copyOf(plan.getOutputMetricsList());
    }
  }

  @Override
  public ImmutableSet<OutputMetric> getClientOutputMetrics() {
    return outputMetrics;
  }

  public ServerPhase getServerPhase() {
    return this.phase.getServerPhase();
  }

  public PhaseSession createPhaseSession(Optional<ByteString> checkpoint, Optional<AppFiles> appFiles) {
    if (serverGraph == null) {
      throw new IllegalStateException(
          "Cannot create more than one phase session. Has createPhaseSession been called already?");
    }
    if (!phase.hasServerPhase()) {
      throw new IllegalStateException("The plan does not have a ServerPhase.");
    }

    AppFiles fileCache = appFiles.orElse(new AppFiles("/tmp"));
    TensorflowPhaseSession phaseSession;
    if (checkpoint.isEmpty()) {
      phaseSession = new TensorflowPhaseSession(this, serverGraph, fileCache);
    } else {
      phaseSession = new TensorflowPhaseSession(this, serverGraph, checkpoint.get(), fileCache);
    }


    // Only one phase session may ever be created from a plan session. Since serverGraph tends
    // to be fairly large and is no longer needed after creating the phase session object above,
    // it makes sense to get it GC'd as soon as possible.
    serverGraph = null;
    return phaseSession;
  }
}