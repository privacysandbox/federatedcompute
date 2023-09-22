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

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertNotNull;

import com.google.protobuf.ByteString;
import java.util.Map;
import java.util.Optional;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class TensorflowPlanSessionTest {

  @Test
  public void testCreatePhaseSessionSingleGradient() throws Exception {
    ByteString checkpoint = ByteString.copyFrom(
        getClass().getResourceAsStream("/com/google/fcp/testdata/init_checkpoint.ckp")
            .readAllBytes());
    ByteString gradient = ByteString.copyFrom(
        getClass().getResourceAsStream("/com/google/fcp/testdata/gradient.ckp").readAllBytes());
    ByteString plan = ByteString.copyFrom(
        getClass().getResourceAsStream("/com/google/fcp/testdata/plan.pb").readAllBytes());

    // Create tensorflow session
    PlanSession planSession = new TensorflowPlanSession(plan);
    PhaseSession phaseSession = planSession.createPhaseSession(Optional.empty());

    // Perform aggregation with gradient
    phaseSession.accumulateClientUpdate(gradient);

    // Finalize aggregation
    ByteString result = phaseSession.toIntermediateUpdate();

    // Assert result
    assertArrayEquals(
        getClass().getResourceAsStream("/com/google/fcp/testdata/intermediate_checkpoint.ckp")
            .readAllBytes(), result.toByteArray());

    // Create tensorflow session
    PlanSession updateSession = new TensorflowPlanSession(plan);
    PhaseSession updatePhaseSession = updateSession.createPhaseSession(Optional.of(checkpoint));

    // Apply update
    updatePhaseSession.accumulateIntermediateUpdate(result);
    updatePhaseSession.applyAggregatedUpdates();
    ByteString newCheckpoint = updatePhaseSession.toCheckpoint();

    // Produce Metrics
    Map<String, Double> metrics = updatePhaseSession.getMetrics();
    assertNotNull(metrics.get("server/client_work/train/binary_accuracy"));
    assertNotNull(metrics.get("server/client_work/train/binary_crossentropy"));
    assertNotNull(metrics.get("server/client_work/train/recall"));
    assertNotNull(metrics.get("server/client_work/train/precision"));
    assertNotNull(metrics.get("server/client_work/train/auc-roc"));
    assertNotNull(metrics.get("server/client_work/train/auc-pr"));

    // Produce server checkpoint
    assertArrayEquals(
        getClass().getResourceAsStream("/com/google/fcp/testdata/expected_checkpoint.ckp")
            .readAllBytes(), newCheckpoint.toByteArray());

    // Produce client checkpoint
    assertNotNull(updatePhaseSession.getClientCheckpoint(Optional.empty()));

    // Clean native resources
    updatePhaseSession.close();
    phaseSession.close();
  }

  @Test
  public void testCreatePhaseSessionSingleSession() throws Exception {
    ByteString checkpoint = ByteString.copyFrom(
        getClass().getResourceAsStream("/com/google/fcp/testdata/init_checkpoint.ckp")
            .readAllBytes());
    ByteString gradient = ByteString.copyFrom(
        getClass().getResourceAsStream("/com/google/fcp/testdata/gradient.ckp").readAllBytes());
    ByteString plan = ByteString.copyFrom(
        getClass().getResourceAsStream("/com/google/fcp/testdata/plan.pb").readAllBytes());

    // Create tensorflow session
    PlanSession planSession = new TensorflowPlanSession(plan);
    PhaseSession phaseSession = planSession.createPhaseSession(Optional.of(checkpoint));

    // Perform aggregation with gradient
    phaseSession.accumulateClientUpdate(gradient);

    // Finalize aggregation
    ByteString result = phaseSession.toIntermediateUpdate();

    // Assert result
    assertArrayEquals(
        getClass().getResourceAsStream("/com/google/fcp/testdata/intermediate_checkpoint.ckp")
            .readAllBytes(), result.toByteArray());

    // Apply update
    phaseSession.accumulateIntermediateUpdate(result);
    phaseSession.applyAggregatedUpdates();
    ByteString newCheckpoint = phaseSession.toCheckpoint();

    assertArrayEquals(
        getClass().getResourceAsStream("/com/google/fcp/testdata/expected_checkpoint.ckp")
            .readAllBytes(), newCheckpoint.toByteArray());

    // Produce Metrics
    Map<String, Double> metrics = phaseSession.getMetrics();
    assertNotNull(metrics.get("server/client_work/train/binary_accuracy"));
    assertNotNull(metrics.get("server/client_work/train/binary_crossentropy"));
    assertNotNull(metrics.get("server/client_work/train/recall"));
    assertNotNull(metrics.get("server/client_work/train/precision"));
    assertNotNull(metrics.get("server/client_work/train/auc-roc"));
    assertNotNull(metrics.get("server/client_work/train/auc-pr"));

    // Produce client checkpoint
    assertNotNull(phaseSession.getClientCheckpoint(Optional.empty()));

    // Produce server checkpoint
    assertArrayEquals(
        getClass().getResourceAsStream("/com/google/fcp/testdata/expected_checkpoint.ckp")
            .readAllBytes(), newCheckpoint.toByteArray());


    // Clean native resources
    phaseSession.close();
  }

  @Test
  public void testCreatePhaseSessionMultipleGradient() throws Exception {
    ByteString checkpoint = ByteString.copyFrom(
        getClass().getResourceAsStream("/com/google/fcp/testdata/init_checkpoint.ckp")
            .readAllBytes());
    ByteString gradient = ByteString.copyFrom(
        getClass().getResourceAsStream("/com/google/fcp/testdata/gradient.ckp").readAllBytes());
    ByteString plan = ByteString.copyFrom(
        getClass().getResourceAsStream("/com/google/fcp/testdata/plan.pb").readAllBytes());

    // Create tensorflow session
    PlanSession planSession = new TensorflowPlanSession(plan);
    PhaseSession phaseSession = planSession.createPhaseSession(Optional.of(checkpoint));

    // Perform aggregation with gradient
    for (int i = 0; i < 10; i++) {
      phaseSession.accumulateClientUpdate(gradient);
    }

    // Finalize aggregation
    ByteString result = phaseSession.toIntermediateUpdate();

    // Assert result
    assertArrayEquals(
        getClass().getResourceAsStream("/com/google/fcp/testdata/intermediate_10x_checkpoint.ckp")
            .readAllBytes(), result.toByteArray());

    // Apply update
    phaseSession.accumulateIntermediateUpdate(result);
    phaseSession.applyAggregatedUpdates();
    ByteString newCheckpoint = phaseSession.toCheckpoint();

    assertArrayEquals(
        getClass().getResourceAsStream("/com/google/fcp/testdata/expected_10x_checkpoint.ckp")
            .readAllBytes(), newCheckpoint.toByteArray());

    // Clean native resources
    phaseSession.close();
  }
}
