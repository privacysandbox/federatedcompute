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
package com.google.fcp.aggregation;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.fcp.tensorflow.NativeHandle;
import com.google.internal.federated.plan.Plan;
import com.google.internal.federated.plan.ServerAggregationConfig;
import com.google.internal.federated.plan.ServerPhaseV2;
import com.google.protobuf.InvalidProtocolBufferException;
import fcp.aggregation.Configuration;
import fcp.aggregation.Tensor;
import java.io.Closeable;
import java.util.List;
import org.tensorflow.framework.TensorShapeProto;
import tensorflow.Struct;

/** A simple wrapper around the checkpoint aggregator. */
public final class AggregationSession implements Closeable {

  static {
    System.loadLibrary("aggregation-jni");
  }

  private final NativeHandle sessionHandle;
  private final byte[] configuration;

  private AggregationSession(long handle, byte[] configuration) {
    checkArgument(handle != 0);
    this.sessionHandle = new NativeHandle(handle);
    this.configuration = configuration;
  }

  // Exception handlers that catch AggregationException should call this method in order to convert
  // them to a generic IllegalStateException.
  private static IllegalStateException onAggregationException(AggregationException e) {
    return new IllegalStateException(
        "An internal exception occurred within the native aggregation session", e);
  }

  /** Creates a new session, based on the plan. */
  public static AggregationSession createFromByteArray(byte[] plan) {
    try {
      byte[] configuration = createConfiguration(plan).toByteArray();
      return new AggregationSession(createNativeFromByteArray(configuration), configuration);
    } catch (AggregationException e) {
      throw onAggregationException(e);
    }
  }

  /** Accumulates the list of checkpoint via nested tensor aggregators in memory. */
  public void accumulate(byte[][] checkpoints) {
    try (NativeHandle.ScopedHandle scopedHandle = sessionHandle.acquire()) {
      runAccumulate(scopedHandle.get(), checkpoints);
    } catch (AggregationException e) {
      throw onAggregationException(e);
    }
  }

  /** Merges the list of serialized aggregators in memory. */
  public void mergeWith(byte[][] serialized) {
    try (NativeHandle.ScopedHandle scopedHandle = sessionHandle.acquire()) {
      mergeWith(scopedHandle.get(), configuration, serialized);
    } catch (AggregationException e) {
      throw onAggregationException(e);
    }
  }

  /** Serialized aggregator in memory to a byte[]. */
  public byte[] serialize() {
    try (NativeHandle.ScopedHandle scopedHandle = sessionHandle.acquire()) {
      return serialize(scopedHandle.get());
    } catch (AggregationException e) {
      throw onAggregationException(e);
    }
  }

  /** Builds a report from the session. */
  public byte[] report() {
    try (NativeHandle.ScopedHandle scopedHandle = sessionHandle.acquire()) {
      return runReport(scopedHandle.get());
    } catch (AggregationException e) {
      throw onAggregationException(e);
    }
  }

  /** Safety net finalizer for cleanup of the wrapped native resource. */
  @Override
  protected void finalize() throws Throwable {
    try {
      if (sessionHandle.isValid()) {
        // Native session was not yet released so release it and log warning.
        close();
      }
    } finally {
      super.finalize();
    }
  }

  /**
   * Closes the session, releasing resources. This must be run in the same thread as create.
   *
   * @throws IllegalStateException with a wrapped AggregationException if closing was not
   *     successful.
   */
  @Override
  public void close() {
    if (!sessionHandle.isValid()) {
      return;
    }
    try (NativeHandle.ScopedHandle scopedHandle = sessionHandle.acquire()) {
      closeNative(scopedHandle.release());
    } catch (AggregationException e) {
      throw onAggregationException(e);
    }
  }

  private static Configuration createConfiguration(byte[] planBytes) {
    Plan plan;
    try {
      plan = Plan.parseFrom(planBytes);
    } catch (InvalidProtocolBufferException e) {
      throw new IllegalArgumentException("Failed to decode plan.");
    }

    if (plan.getPhaseList().isEmpty()) {
      throw new IllegalArgumentException("Found plan with no phases.");
    }

    if (!plan.getPhase(0).hasServerPhaseV2()) {
      throw new IllegalArgumentException("Found plan without ServerPhaseV2 message.");
    }

    ServerPhaseV2 phase = plan.getPhase(0).getServerPhaseV2();
    List<ServerAggregationConfig> serverAggregationConfigs = phase.getAggregationsList();
    List<Configuration.IntrinsicConfig> intrinsicConfigs =
        serverAggregationConfigs.stream()
            .map(AggregationSession::convertIntrinsicConfig)
            .collect(toImmutableList());
    return Configuration.newBuilder().addAllIntrinsicConfigs(intrinsicConfigs).build();
  }

  private static Tensor.TensorSpecProto convertTensorSpecProto(Struct.TensorSpecProto from) {
    Tensor.TensorSpecProto.Builder to = Tensor.TensorSpecProto.newBuilder();
    switch (from.getDtype()) {
      case DT_INVALID:
        to.setDtype(Tensor.DataType.DT_INVALID);
        break;
      case DT_FLOAT:
        to.setDtype(Tensor.DataType.DT_FLOAT);
        break;
      case DT_DOUBLE:
        to.setDtype(Tensor.DataType.DT_DOUBLE);
        break;
      case DT_INT32:
        to.setDtype(Tensor.DataType.DT_INT32);
        break;
      case DT_STRING:
        to.setDtype(Tensor.DataType.DT_STRING);
        break;
      case DT_INT64:
        to.setDtype(Tensor.DataType.DT_INT64);
        break;
      case DT_UINT64:
        to.setDtype(Tensor.DataType.DT_UINT64);
        break;
      default:
        throw new UnsupportedOperationException(
            "Datatype " + from.getDtype() + " is not supported by Aggregation Service.");
    }
    to.getShapeBuilder()
        .addAllDimSizes(
            from.getShape().getDimList().stream()
                .map(TensorShapeProto.Dim::getSize)
                .collect(toImmutableList()));
    to.setName(from.getName());
    return to.build();
  }

  private static Configuration.IntrinsicConfig.IntrinsicArg convertIntrinsicArg(
      ServerAggregationConfig.IntrinsicArg from) {
    Configuration.IntrinsicConfig.IntrinsicArg.Builder to =
        Configuration.IntrinsicConfig.IntrinsicArg.newBuilder();
    if (from.hasStateTensor()) {
      throw new UnsupportedOperationException(
          "`parameter` arg is not currently support by Aggregation Service.");
    }
    to.setInputTensor(convertTensorSpecProto(from.getInputTensor()));
    return to.build();
  }

  private static Configuration.IntrinsicConfig convertIntrinsicConfig(
      ServerAggregationConfig from) {
    Configuration.IntrinsicConfig.Builder to = Configuration.IntrinsicConfig.newBuilder();
    to.addAllInnerIntrinsics(
        from.getInnerAggregationsList().stream()
            .map(AggregationSession::convertIntrinsicConfig)
            .collect(toImmutableList()));
    to.addAllOutputTensors(
        from.getOutputTensorsList().stream()
            .map(AggregationSession::convertTensorSpecProto)
            .collect(toImmutableList()));
    to.addAllIntrinsicArgs(
        from.getIntrinsicArgsList().stream()
            .map(AggregationSession::convertIntrinsicArg)
            .collect(toImmutableList()));
    to.setIntrinsicUri(from.getIntrinsicUri());
    return to.build();
  }

  // Native API
  // ==========

  /** Starts a session based on the given configuration, returning a handle for it. */
  static native long createNativeFromByteArray(byte[] configuration) throws AggregationException;

  // CAREFUL: don't make the following native calls static because it can cause a race condition
  // between the native execution and the object finalize() call.

  /** Closes the session. The handle is not usable afterwards. */
  native void closeNative(long session) throws AggregationException;

  /** Accumulates the provided checkpoint using the native session. */
  native void runAccumulate(long session, byte[][] checkpoints) throws AggregationException;

  /** Merges the serialized aggregator using the native session. */
  native void mergeWith(long session, byte[] configuration, byte[][] serialized)
      throws AggregationException;

  /** Serializes the internal state of the aggregator using the native session. */
  native byte[] serialize(long session) throws AggregationException;

  /** Creates a report using the native session. */
  native byte[] runReport(long session) throws AggregationException;
}
