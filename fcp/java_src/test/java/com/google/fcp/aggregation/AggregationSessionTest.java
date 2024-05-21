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

import com.google.fcp.plan.PhaseSessionV2;
import com.google.fcp.plan.TensorflowPhaseSessionV2;
import com.google.protobuf.ByteString;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.tensorflow.framework.TensorProto;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

@RunWith(JUnit4.class)
public class AggregationSessionTest {

  private byte[] accumulateIntermediate(byte[] plan_bytes, byte[] gradient) throws Exception {
    AggregationSession ag_session = AggregationSession.createFromByteArray(plan_bytes);
    try {
      ag_session.accumulate(new byte[][] {gradient, gradient});
      return ag_session.serialize();
    } finally {
      ag_session.close();
    }
  }

  @Test
  public void testReportAggregationSession() throws Exception {
    byte[] plan_bytes =
        getClass().getResourceAsStream("/com/google/fcp/testdata/keras_plan_v2.pb").readAllBytes();
    byte[] checkpoint =
        getClass()
            .getResourceAsStream("/com/google/fcp/testdata/keras_checkpoint_v2.ckp")
            .readAllBytes();
    byte[] gradient =
        getClass()
            .getResourceAsStream("/com/google/fcp/testdata/keras_gradient_v2.ckp")
            .readAllBytes();

    byte[] intermediate = accumulateIntermediate(plan_bytes, gradient);

    AggregationSession ag_session = AggregationSession.createFromByteArray(plan_bytes);
    ag_session.mergeWith(new byte[][] {intermediate, intermediate});
    byte[] aggregateResult = ag_session.report();
    assertTrue(aggregateResult.length > 0);
    ag_session.close();

    PhaseSessionV2 phaseSessionV2 = new TensorflowPhaseSessionV2(ByteString.copyFrom(plan_bytes));
    PhaseSessionV2.IntermediateResult intermediateResult = phaseSessionV2.getClientCheckpoint(ByteString.copyFrom(checkpoint));
    PhaseSessionV2.Result result = phaseSessionV2.getResult(ByteString.copyFrom(aggregateResult), intermediateResult.interemdiateState());

    ByteString newCheckpoint = result.updatedServerState();
    assertEquals(true, newCheckpoint.size() > 0);

    // Produce Metrics
    Map<String, TensorProto> metrics = result.metrics();
    assertEquals(
        0.4957627, metrics.get("server/client_work/train/precision").getFloatVal(0), 0.000001);
    assertEquals(
        0.47855276, metrics.get("server/client_work/train/auc-roc").getFloatVal(0), 0.000001);
    assertEquals(
        0.51270926, metrics.get("server/client_work/train/auc-pr").getFloatVal(0), 0.000001);
    assertEquals(
        0.46931407,
        metrics.get("server/client_work/train/binary-accuracy").getFloatVal(0),
        0.000001);
    assertEquals(
        0.7109273,
        metrics.get("server/client_work/train/binary-cross-entropy").getFloatVal(0),
        0.000001);
    assertEquals(
        0.80689657, metrics.get("server/client_work/train/recall").getFloatVal(0), 0.000001);
    assertEquals(0.71106434, metrics.get("server/client_work/train/loss").getFloatVal(0), 0.000001);
    assertEquals(0, metrics.get("server/finalizer/update_non_finite").getIntVal(0), 0.000001);

    ByteString clientCheckpoint =
        new TensorflowPhaseSessionV2(ByteString.copyFrom(plan_bytes))
            .getClientCheckpoint(newCheckpoint).clientCheckpoint();
    assertTrue(clientCheckpoint.size() > 0);
  }

  @Test
  public void testInvalidPlan() {
    assertThrows(
        IllegalArgumentException.class,
        () -> AggregationSession.createFromByteArray(new byte[] {1, 2}));
  }

  @Test
  public void testPlanV1() throws Exception {
    byte[] plan_bytes =
        getClass().getResourceAsStream("/com/google/fcp/testdata/plan.pb").readAllBytes();
    assertThrows(
        IllegalArgumentException.class, () -> AggregationSession.createFromByteArray(plan_bytes));
  }

  @Test
  public void testAccumulateBadGradient() throws Exception {
    byte[] plan_bytes =
        getClass().getResourceAsStream("/com/google/fcp/testdata/keras_plan_v2.pb").readAllBytes();
    assertThrows(
        IllegalStateException.class, () -> accumulateIntermediate(plan_bytes, new byte[] {1, 2}));
  }

  @Test
  public void testAccumulateAlreadyClosed() throws Exception {
    byte[] plan_bytes =
        getClass().getResourceAsStream("/com/google/fcp/testdata/keras_plan_v2.pb").readAllBytes();
    byte[] gradient =
        getClass()
            .getResourceAsStream("/com/google/fcp/testdata/keras_gradient_v2.ckp")
            .readAllBytes();
    AggregationSession ag_session = AggregationSession.createFromByteArray(plan_bytes);
    ag_session.close();
    assertThrows(IllegalStateException.class, () -> ag_session.accumulate(new byte[][] {gradient}));
  }

  @Test
  public void testReportAlreadyClosed() throws Exception {
    byte[] plan_bytes =
        getClass().getResourceAsStream("/com/google/fcp/testdata/keras_plan_v2.pb").readAllBytes();
    byte[] gradient =
        getClass()
            .getResourceAsStream("/com/google/fcp/testdata/keras_gradient_v2.ckp")
            .readAllBytes();
    AggregationSession ag_session = AggregationSession.createFromByteArray(plan_bytes);
    ag_session.accumulate(new byte[][]{gradient});
    ag_session.close();
    assertThrows(IllegalStateException.class, () -> ag_session.report());
  }

  @Test
  public void testReportTwice() throws Exception {
    byte[] plan_bytes =
        getClass().getResourceAsStream("/com/google/fcp/testdata/keras_plan_v2.pb").readAllBytes();
    AggregationSession ag_session = AggregationSession.createFromByteArray(plan_bytes);
    ag_session.report();
    assertThrows(IllegalStateException.class, () -> ag_session.report());
  }
}
