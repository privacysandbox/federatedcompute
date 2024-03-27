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

import com.google.fcp.tensorflow.AppFiles;
import com.google.internal.federated.plan.OutputMetric;
import com.google.internal.federated.plan.ServerPhase;
import com.google.protobuf.ByteString;
import java.util.Optional;
import java.util.Set;

/**
 * Objects which implement this interface encapsulate the selection of a specific execution plan
 * engine and a specific execution plan. Such an object has the code and data necessary to create
 * PhaseSession objects, and to serve the client execution plan to clients.
 *
 * <p>Typically, implementations of this object will encapsulate the parsed form of an execution
 * plan, in a form that is efficient for the plan engine.
 */
public interface PlanSession {

  /**
   * Returns the set of output metrics that are defined for a given execution. The returned list is
   * immutable.
   */
  Set<OutputMetric> getClientOutputMetrics();

  /**
   * Returns a new phase session, which has been initialized using the given checkpoint if
   * available.
   *
   * @throws {@link IllegalStateException} if a phase session has already been created or a
   *     ServerPhase does not exist.
   */
  PhaseSession createPhaseSession(Optional<ByteString> checkpoint, Optional<AppFiles> appFiles);

  /**
   * Retrieves the plan's server phase, which represents the plan for aggregating multiple client
   * updates.
   */
  ServerPhase getServerPhase();
}