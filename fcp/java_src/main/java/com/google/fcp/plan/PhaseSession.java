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

import com.google.protobuf.ByteString;
import java.util.Map;
import java.util.Optional;
import org.tensorflow.framework.TensorProto;

/**
 * Describes an interface to the state of a population.
 *
 * <p>A state may be defunct, which happens if it is succeeded by a newer state. Some operations
 * fail on defunct states, others have no effects.
 */
public interface PhaseSession {

  /**
   * Accumulates the given client update into in-memory state.
   *
   * @throws {@link IllegalStateException} Generic exception thrown for tensorflow related
   *     exceptions.
   */
  void accumulateClientUpdate(ByteString clientUpdate);

  /**
   * Accumulates the given intermediate update into in-memory state.
   *
   * @throws {@link IllegalStateException} Generic exception thrown for tensorflow related
   *     exceptions.
   */
  void accumulateIntermediateUpdate(ByteString aggregatedUpdate);

  /**
   * Applies the aggregated updates to the current session, finalizing the round.
   *
   * @throws {@link IllegalStateException} Generic exception thrown for tensorflow related
   *     exceptions.
   */
  void applyAggregatedUpdates();

  /**
   * Exports serialized checkpoint containing all the accumulations in a form ready to be written to
   * stable storage.
   *
   * <p>The publicly-visible state of the object is not changed.
   *
   * @throws {@link IllegalStateException} Generic exception thrown for tensorflow related
   *     exceptions.
   */
  ByteString toCheckpoint();

  /**
   * Exports serialized aggregated update containing all client updates.
   *
   * <p>The publicly-visible state of the object is not changed.
   *
   * @throws {@link IllegalStateException} Generic exception thrown for tensorflow related
   *     exceptions.
   */
  ByteString toIntermediateUpdate();

  /**
   * Extracts a set of metrics in double value from a session. The set of metrics are defined in the
   * execution plan, and are dependent on the current phase index. The returned map is immutable.
   *
   * @throws {@link IllegalStateException} Generic exception thrown for tensorflow related
   *     exceptions.
   */
  Map<String, Double> getMetrics();

  /**
   * Extracts a set of metrics in tensor value from a session. The set of metrics are defined in the
   * execution plan, and are dependent on the current phase index. The returned map is immutable.
   *
   * @throws {@link IllegalStateException} Generic exception thrown for tensorflow related
   *     exceptions.
   */
  Map<String, TensorProto> getTensorValueMetrics();

  /**
   * Retrieves an opaque representation of the checkpoint to be passed to clients. The checkpoint
   * represents the state from which the client starts.
   *
   * <p>The PhaseSession must not be closed.
   *
   * @param sessionToken allows TensorFlow ops such as `ServeSlices` to refer to callbacks
   *     registered before running the session.
   * @throws {@link IllegalStateException} Generic exception thrown for tensorflow related
   *     exceptions.
   */
  ByteString getClientCheckpoint(Optional<byte[]> sessionToken);

  /**
   * Releases all resources held by this object (such as unmanaged / C++ resources). After this
   * call, no other methods defined on this interface should be called.
   */
  void close();
}