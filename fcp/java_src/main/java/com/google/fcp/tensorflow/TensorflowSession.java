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
package com.google.fcp.tensorflow;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;
import java.io.Closeable;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.charset.StandardCharsets;
import java.util.Collection;
import java.util.Map;
import org.tensorflow.framework.DataType;
import org.tensorflow.framework.TensorProto;
import org.tensorflow.framework.TensorShapeProto;
import org.tensorflow.util.SaverDef;

/**
 * A simple wrapper around the Tensorflow engine, for the standard Java proto model. Uses
 * {@link TensorProto} to represent tensors.
 */
public final class TensorflowSession implements Closeable {

  static final String[] EMPTY_STRING_ARRAY = new String[0];
  /**
   * Maximum permitted size of a Tensor in a checkpoint, in bytes.
   */
  private static final int MAX_TENSOR_SIZE = 500 * 1024 * 1024;

  private static final Charset UTF_8 = StandardCharsets.UTF_8;

  static {
    System.loadLibrary("tensorflow-jni");
  }

  private final AppFiles fileCache;
  private final NativeHandle sessionHandle;

  private TensorflowSession(AppFiles fileCache, long handle) {
    this.fileCache = checkNotNull(fileCache);
    checkArgument(handle != 0);
    this.sessionHandle = new NativeHandle(handle);
  }

  /**
   * Creates a new session, based on the serialized GraphDef found in the byte array. This method is
   * provided in order to support situations where performance is important, and the cost of
   * allocating / copying the byte[] from ByteString is unacceptable. There is currently no
   * efficient way of passing ByteString to C++ code, and TensorflowSession is implemented by
   * calling into the C++ TensorFlow implementation.
   */
  public static TensorflowSession createFromByteArray(AppFiles fileCache, byte[] graphDef)
      throws TensorflowException {
    return new TensorflowSession(fileCache, createNativeFromByteArray(graphDef));
  }

  /**
   * Helper method that converts an array of tensor names and an array of corresponding serialized
   * TensorProtos into a map.
   */
  private static ImmutableMap<String, TensorProto> convertToTensorMap(
      String[] tensorNames, byte[][] tensorValues) throws TensorflowException {
    Preconditions.checkArgument(tensorNames.length == tensorValues.length,
        "Length of tensorNames does not match tensorValues");
    if (tensorNames.length == 0) {
      return ImmutableMap.of();
    } else {
      ImmutableMap.Builder<String, TensorProto> result =
          ImmutableMap.builderWithExpectedSize(tensorNames.length);
      for (int i = 0; i < tensorNames.length; i++) {
        byte[] tensorData = tensorValues[i];
        if (tensorData == null) {
          throw new TensorflowException(
              String.format("Tensorflow run did not write output '%s'", tensorNames[i]));
        }
        try {
          result.put(
              tensorNames[i],
              TensorProto.parseFrom(tensorData));
        } catch (InvalidProtocolBufferException e) {
          throw new TensorflowException("Invalid proto output from tensorflow", e);
        }
      }
      return result.buildOrThrow();
    }
  }

  /**
   * Helper to create a tensor proto which represents a single string scalar.
   */
  public static TensorProto stringTensor(String value) {
    return TensorProto.newBuilder()
        .setDtype(DataType.DT_STRING)
        .addStringVal(ByteString.copyFrom(value, UTF_8))
        .setTensorShape(TensorShapeProto.getDefaultInstance())
        .build();
  }

  /**
   * Helper to create a tensor proto which represents a single string scalar.
   */
  public static TensorProto stringTensor(byte[] value) {
    return TensorProto.newBuilder()
        .setDtype(DataType.DT_STRING)
        .addStringVal(ByteString.copyFrom(value))
        .setTensorShape(TensorShapeProto.getDefaultInstance())
        .build();
  }

  /**
   * Safety net finalizer for cleanup of the wrapped native resource.
   */
  @Override
  protected void finalize() throws Throwable {
    try {
      if (sessionHandle.isValid()) {
        // Native tensorflow session was not yet released so release it and log warning.
        close();
      }
    } finally {
      super.finalize();
    }
  }

  /**
   * Closes the session, releasing resources. This must be run in the same thread as create.
   *
   * @throws IllegalStateException with a wrapped TensorflowException if closing was not
   *     successful. This can happen if other threads are still running the graph in this session.
   */
  @Override
  public void close() {
    if (!sessionHandle.isValid()) {
      return;
    }
    try (NativeHandle.ScopedHandle scopedHandle = sessionHandle.acquire()) {
      closeNative(scopedHandle.release());
    } catch (TensorflowException e) {
      throw new IllegalStateException(e);
    }
  }

  /**
   * Runs the graph of this session. This is thread-safe.
   *
   * @param inputs an optional map from tensor names to tensor values.
   * @param outputNames an optional list of output tensor names.
   * @param targetNodeNames an optional list of nodes which do not produce output.
   * @return a map of tensor names to protos for produced outputs. Always non-null.
   * @throws TensorflowException if something goes wrong.
   */
  public Map<String, TensorProto> run(
      Map<String, TensorProto> inputs,
      Collection<String> outputNames,
      Collection<String> targetNodeNames)
      throws TensorflowException {
    int inputCount = inputs == null ? 0 : inputs.size();
    String[] inputNamesArray = new String[inputCount];
    byte[][] inputValuesArray = new byte[inputCount][];
    if (inputCount > 0) {
      int i = 0;
      for (Map.Entry<String, TensorProto> entry : inputs.entrySet()) {
        inputNamesArray[i] = entry.getKey();
        inputValuesArray[i] = entry.getValue().toByteArray();
        i++;
      }
    }
    String[] outputNamesArray =
        outputNames == null ? EMPTY_STRING_ARRAY : outputNames.toArray(EMPTY_STRING_ARRAY);
    String[] targetNodesArray;
    if (targetNodeNames == null) {
      targetNodesArray = EMPTY_STRING_ARRAY;
    } else {
      targetNodesArray = new String[targetNodeNames.size()];
      int i = 0;
      for (String name : targetNodeNames) {
        if (name.endsWith(":0")) {
          // Py wrappers often produce a tensor name here instead of a node name,
          // which the native session doesn't like. Remove the trailing ':0'
          // to work around this.
          name = name.substring(0, name.length() - 2);
        }
        targetNodesArray[i++] = name;
      }
    }
    byte[][] outputValuesArray = new byte[outputNamesArray.length][];
    try (NativeHandle.ScopedHandle scopedHandle = sessionHandle.acquire()) {
      runNative(
          scopedHandle.get(),
          inputNamesArray,
          inputValuesArray,
          outputNamesArray,
          outputValuesArray,
          targetNodesArray);
    } catch (IllegalStateException e) {
      throw new TensorflowException("TensorflowSession was already closed", e);
    }

    return convertToTensorMap(outputNamesArray, outputValuesArray);
  }

  /**
   * Shortcut for running the graph in the case no outputs need to be produced.
   *
   * @param inputs an optional map from tensor names to tensor values.
   * @param targetNodeNames an optional list of nodes which do not produce output.
   * @throws TensorflowException if something goes wrong.
   */
  public void run(Map<String, TensorProto> inputs, String... targetNodeNames)
      throws TensorflowException {
    run(inputs, null, ImmutableList.copyOf(targetNodeNames));
  }

  /**
   * Shortcut for running a single target only if it is not null and not empty.
   */
  public void maybeRun(String targetNodeName) throws TensorflowException {
    maybeRun(null, targetNodeName);
  }

  /**
   * Shortcut for running a single target only if it is not null and not empty.
   *
   * @param inputs an optional map from tensor names to tensor values.
   */
  public void maybeRun(Map<String, TensorProto> inputs, String targetNodeName)
      throws TensorflowException {
    if (targetNodeName != null && !targetNodeName.isEmpty()) {
      run(inputs, targetNodeName);
    }
  }

  /**
   * Saves the current session state into a byte string.
   */
  public ByteString saveState(SaverDef saverDef) throws TensorflowException {
    return saveState(
        saverDef.getFilenameTensorName(), saverDef.getSaveTensorName(), ImmutableMap.of());
  }

  /**
   * Saves the current session state into a byte string.
   */
  public ByteString saveState(SaverDef saverDef, String sessionTokenTensorName, byte[] sessionToken)
      throws TensorflowException {
    return saveState(
        saverDef.getFilenameTensorName(),
        saverDef.getSaveTensorName(),
        ImmutableMap.of(sessionTokenTensorName, stringTensor(sessionToken)));
  }

  /**
   * Saves the current session state to a file taking an optional map of inputs that is passed to
   * the TensorFlow session.
   */
  private void saveState(
      String filenameTensorName, String filename, String saveOp, Map<String, TensorProto> inputs)
      throws TensorflowException {
    if (saveOp.endsWith(":0")) {
      saveOp = saveOp.substring(0, saveOp.length() - 2);
    }

    ImmutableMap.Builder<String, TensorProto> combinedInputs = ImmutableMap.builder();
    combinedInputs.put(filenameTensorName, stringTensor(filename));
    combinedInputs.putAll(inputs);

    run(combinedInputs.buildOrThrow(), null, ImmutableList.of(saveOp));
  }

  /**
   * Saves the current session state into a ByteString taking an optional map of inputs that is
   * passed to the TensorFlow session.
   */
  private ByteString saveState(
      String filenameTensorName, String saveOp, Map<String, TensorProto> inputs)
      throws TensorflowException {
    // There is no API to do this natively, so we do have to create a temp file.
    try {
      File tempFile = fileCache.createTempFile("checkpoint", ".ckp");
      try {
        saveState(filenameTensorName, tempFile.getPath(), saveOp, inputs);
        try (final FileInputStream stream = new FileInputStream(tempFile)) {
          return FileUtil.readByteString(stream);
        }
      } finally {
        tempFile.delete();
      }
    } catch (IOException e) {
      throw new TensorflowException("cannot save state", e);
    }
  }

  /**
   * Loads session state from a file based on the given SaverDef.
   *
   * @throws TensorflowException if something goes wrong
   */
  public void loadState(SaverDef saverDef, String fileName) throws TensorflowException {
    validateCheckpoint(fileName, MAX_TENSOR_SIZE);
    run(
        ImmutableMap.of(saverDef.getFilenameTensorName(), stringTensor(fileName)),
        null,
        ImmutableList.of(saverDef.getRestoreOpName()));
  }

  /**
   * Returns a newly created temporary checkpoint file with the given state as content.
   */
  public File writeTempCheckpointFile(ByteString state) throws TensorflowException {
    File tempFile = null;
    try {
      tempFile = fileCache.createTempFile("checkpoint", ".ckp");
      try (final FileOutputStream stream = new FileOutputStream(tempFile)) {
        FileUtil.writeByteString(state, stream);
      }
      return tempFile;
    } catch (IOException e) {
      if (tempFile != null) {
        tempFile.delete();
      }
      throw new TensorflowException("Failed to write a temporary checkpoint", e);
    }
  }
  // Native API
  // ==========

  /**
   * Starts a session based on the given serialized client graph, returning a handle for it.
   */
  static native long createNativeFromByteArray(byte[] clientDef) throws TensorflowException;

  // CAREFUL: don't make the following native calls static because it can cause a race condition
  // between the native execution and the object finalize() call.

  /**
   * Closes the session. The handle is not usable afterwards.
   */
  native void closeNative(long session) throws TensorflowException;

  /**
   * Runs the graph.
   *
   * @param inputTensorNames the names of the input tensors.
   * @param inputTensorProtos the values (TensorT) for the inputs. Must have same length as
   *     inputTensorNames.
   * @param outputTensorNames the names the output tensors.
   * @param outputTensorProtos an array of same size as outputTensorNames. The output tensor
   *     protos are stored in here.
   * @param targetNodeNames any nodes to run which do not produce output tensors.
   */
  native void runNative(
      long session,
      String[] inputTensorNames,
      byte[][] inputTensorProtos,
      String[] outputTensorNames,
      byte[][] outputTensorProtos,
      String[] targetNodeNames)
      throws TensorflowException;

  /**
   * Validates a TensorFlow checkpoint.
   *
   * @param filepattern the checkpoint's file name
   * @param maxTensorSize the maximum tensor size in bytes to allow
   */
  native void validateCheckpoint(String filepattern, int maxTensorSize) throws TensorflowException;

  /**
   * Retrieves tensors from a TensorFlow checkpoint.
   *
   * @param filepattern the checkpoint's file name
   * @param tensorNames the tensors to retrieve
   * @param tensorProtos an array of the same size as tensorNames. The retrieved tensor protos are
   *     stored in here.
   */
}