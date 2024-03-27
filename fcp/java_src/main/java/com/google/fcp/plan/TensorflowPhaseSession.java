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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.fcp.tensorflow.AppFiles;
import com.google.fcp.tensorflow.TensorflowException;
import com.google.fcp.tensorflow.TensorflowSession;
import com.google.internal.federated.plan.CheckpointOp;
import com.google.internal.federated.plan.Metric;
import com.google.internal.federated.plan.ServerPhase;
import com.google.protobuf.ByteString;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import javax.annotation.Nullable;
import javax.annotation.concurrent.GuardedBy;
import org.tensorflow.framework.TensorProto;

class TensorflowPhaseSession implements PhaseSession {

  private final AppFiles fileCache;

  // Deeply-immutable values
  private final TensorflowPlanSession planSession;

  // If the session is null, this state is defunct (happens after close() call).
  @GuardedBy("this")
  @Nullable
  private TensorflowSession session;

  // Indicates whether accumulate() has ever been called on this state.
  @GuardedBy("this")
  private boolean hasAccumulated;

  TensorflowPhaseSession(
      TensorflowPlanSession planSession, byte[] serverGraph, ByteString checkpoint, AppFiles fileCache) {
    this(planSession, serverGraph, fileCache);
    checkNotNull(checkpoint);
    checkArgument(!checkpoint.isEmpty());

    try {
      restoreState(planSession.serverSavepoint, checkpoint, ImmutableMap.of());
    } catch (TensorflowException e) {
      close();
      throw onTensorflowException(e);
    }
  }

  TensorflowPhaseSession(
      TensorflowPlanSession planSession, byte[] serverGraph, AppFiles fileCache) {
    this.planSession = checkNotNull(planSession);
    this.fileCache = fileCache;
    checkNotNull(serverGraph);

    try {
      this.session = TensorflowSession.createFromByteArray(fileCache, serverGraph);
    } catch (TensorflowException e) {
      close();
      throw onTensorflowException(e);
    }
  }

  // Exception handlers that catch TensorflowException should call this method in order to convert
  // them to a generic IllegalStateException.
  private static IllegalStateException onTensorflowException(TensorflowException e) {
    return new IllegalStateException("An internal exception occurred within TensorFlow", e);
  }

  private static Map<String, TensorProto> addCheckpointPathToOpInputs(
      Map<String, TensorProto> originalOpInputs, CheckpointOp checkpointOp, File checkpointFile) {
    String checkpointTensorName = checkpointOp.getSaverDef().getFilenameTensorName();
    if (checkpointTensorName.isEmpty()) {
      // Saver is not used, there is nothing to do.
      return originalOpInputs;
    }

    // Add extra argument with checkpointPath to the originalOpInputs Map.
    TensorProto checkpointTensor = TensorflowSession.stringTensor(checkpointFile.getPath());
    ImmutableMap.Builder<String, TensorProto> updatedOpInputsBuilder =
        ImmutableMap.builderWithExpectedSize(originalOpInputs.size() + 1);
    updatedOpInputsBuilder.put(checkpointTensorName, checkpointTensor);
    updatedOpInputsBuilder.putAll(originalOpInputs);
    return updatedOpInputsBuilder.buildOrThrow();
  }

  @GuardedBy("this")
  private void checkOpen() {
    Preconditions.checkState(session != null);
  }

  @Override
  public synchronized void close() {
    if (session != null) {
      session.close();
      session = null;
    }
  }

  @Override
  public synchronized void accumulateClientUpdate(ByteString clientUpdate) {
    checkOpen();
    ServerPhase phase = planSession.getServerPhase();
    try {
      if (!hasAccumulated) {
        // Clear accumulators the first time we run accumulation.
        session.maybeRun(phase.getPhaseInitOp());
        hasAccumulated = true;
      }
    } catch (TensorflowException e) {
      throw onTensorflowException(e);
    }

    try {
      // Load client update into session.
      restoreState(phase.getReadUpdate(), clientUpdate, ImmutableMap.of());
    } catch (TensorflowException e) {
      throw onTensorflowException(e);
    }

    try {
      // Accumulate.
      session.maybeRun(phase.getAggregateIntoAccumulatorsOp());
    } catch (TensorflowException e) {
      throw onTensorflowException(e);
    }
  }

  @Override
  public synchronized void accumulateIntermediateUpdate(ByteString aggregatedUpdate) {
    checkOpen();
    ServerPhase phase = planSession.getServerPhase();
    try {
      if (!hasAccumulated) {
        // Clear accumulators the first time we run accumulation.
        session.maybeRun(phase.getPhaseInitOp());
        hasAccumulated = true;
      }

      // Load intermediate update into session.
      restoreState(phase.getReadIntermediateUpdate(), aggregatedUpdate, ImmutableMap.of());

      // Accumulate.
      session.maybeRun(phase.getIntermediateAggregateIntoAccumulatorsOp());
    } catch (TensorflowException e) {
      throw onTensorflowException(e);
    }
  }

  @Override
  public synchronized void applyAggregatedUpdates() {
    checkOpen();
    try {
      // Apply accumulators to global state
      session.maybeRun(planSession.getServerPhase().getApplyAggregratedUpdatesOp());
    } catch (TensorflowException e) {
      throw onTensorflowException(e);
    }
  }

  @Override
  public synchronized ByteString toCheckpoint() {
    checkOpen();
    try {
      // Serialize the checkpoint data and return it.
      return saveState(planSession.serverSavepoint);
    } catch (TensorflowException e) {
      throw onTensorflowException(e);
    }
  }

  @Override
  public synchronized ByteString toIntermediateUpdate() {
    try {
      checkOpen();
      // Serialize the intermediate update and return it.
      return saveState(planSession.getServerPhase().getWriteIntermediateUpdate());
    } catch (TensorflowException e) {
      throw onTensorflowException(e);
    }
  }

  @GuardedBy("this")
  private ByteString saveState(CheckpointOp checkpointOp) throws TensorflowException {
    return saveState(checkpointOp, Optional.empty());
  }

  @GuardedBy("this")
  private ByteString saveState(CheckpointOp checkpointOp, Optional<byte[]> sessionToken)
      throws TensorflowException {
    ImmutableMap<String, TensorProto> beforeAndAfterOpsInputs = null;
    if (sessionToken.isPresent() && !checkpointOp.getSessionTokenTensorName().isEmpty()) {
      // If the plan supports passing the session token and the session token is present,
      // pass it along to the 'before' and 'after' ops, since the session token may be used in
      // either of the 'before', 'after', *or* SaverDef ops.
      beforeAndAfterOpsInputs =
          ImmutableMap.of(
              checkpointOp.getSessionTokenTensorName(),
              TensorflowSession.stringTensor(sessionToken.get()));
    }
    session.maybeRun(beforeAndAfterOpsInputs, checkpointOp.getBeforeSaveOp());

    ByteString result = ByteString.EMPTY;
    if (checkpointOp.hasSaverDef()) {
      if (sessionToken.isPresent() && !checkpointOp.getSessionTokenTensorName().isEmpty()) {
        // If the plan supports passing the session token and the session token is present,
        // pass it along to the SaverDef op.
        result =
            session.saveState(
                checkpointOp.getSaverDef(),
                checkpointOp.getSessionTokenTensorName(),
                sessionToken.get());
      } else {
        result = session.saveState(checkpointOp.getSaverDef());
      }
    }
    session.maybeRun(beforeAndAfterOpsInputs, checkpointOp.getAfterSaveOp());
    return result;
  }

  @Override
  public synchronized Map<String, Double> getMetrics() {
    Map<String, TensorProto> tensorValueMetrics = getTensorValueMetrics();
    return tensorValueMetrics.entrySet().stream()
        .collect(ImmutableMap.toImmutableMap(e -> e.getKey(), e -> getTensorValueAsScalarDouble(e.getValue())));
  }

  @Override
  public synchronized ByteString getClientCheckpoint(Optional<byte[]> sessionToken) {
    checkOpen();
    try {
      return saveState(planSession.getServerPhase().getWriteClientInit(), sessionToken);
    } catch (TensorflowException e) {
      throw onTensorflowException(e);
    }
  }

  @Override
  public synchronized Map<String, TensorProto> getTensorValueMetrics() {
    try {
      checkOpen();

      ServerPhase serverPhase = planSession.getServerPhase();
      List<Metric> metrics = serverPhase.getMetricsList();

      if (!metrics.isEmpty()) {
        ArrayList<String> metricVarNames = new ArrayList<>();
        for (Metric metric : metrics) {
          metricVarNames.add(metric.getVariableName());
        }

        Map<String, TensorProto> tensors = session.run(null, metricVarNames, null);

        ImmutableMap.Builder<String, TensorProto> results = new ImmutableMap.Builder<>();
        for (Metric metric : metrics) {
          TensorProto tensor = tensors.get(metric.getVariableName());
          if (tensor != null) {
            results.put(metric.getStatName(), tensor);
          }
        }

        return results.buildOrThrow();
      } else {
        // This execution plan does not define any server metrics.
        return ImmutableMap.of();
      }
    } catch (TensorflowException e) {
      throw onTensorflowException(e);
    }
  }

  private static double getTensorValueAsScalarDouble(TensorProto tensor) {
    switch (tensor.getDtype()) {
      case DT_FLOAT:
        return tensor.getFloatVal(0);
      case DT_DOUBLE:
        return tensor.getDoubleVal(0);
      case DT_INT32:
        return tensor.getIntVal(0);
      case DT_INT64:
        return tensor.getInt64Val(0);
      case DT_BOOL:
        return tensor.getBoolVal(0) ? 1.0 : 0.0;
      default:
        return 0.0;
    }
  }

  @GuardedBy("this")
  private void restoreState(
      CheckpointOp checkpointOp,
      ByteString checkpoint,
      Map<String, TensorProto> beforeRestoreInputs)
      throws TensorflowException {
    // Create checkpoint so the path can be intercepted by before restore ops.
    File checkpointFile = session.writeTempCheckpointFile(checkpoint);
    try {
      beforeRestoreInputs =
          addCheckpointPathToOpInputs(beforeRestoreInputs, checkpointOp, checkpointFile);

      // Perform the restore sequence
      session.maybeRun(beforeRestoreInputs, checkpointOp.getBeforeRestoreOp());
      if (checkpointOp.hasSaverDef()) {
        session.loadState(checkpointOp.getSaverDef(), checkpointFile.getPath());
      }
      session.maybeRun(checkpointOp.getAfterRestoreOp());
    } finally {
      checkpointFile.delete();
    }
  }
}