/*
 * Copyright 2023 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <jni.h>
#include <stdio.h>
#include <string.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/status/statusor.h"
#include "fcp/jni/jni_util.h"
#include "fcp/jni/more_jni_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/protobuf/struct.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/tensor_slice_reader.h"
#include "validate_checkpoint.h"

using fcp::jni::BinaryProtoToTensor;
using fcp::jni::JStringArrayToStringVector;
using fcp::jni::JStringToString;
using fcp::jni::ParseProtoFromJByteArray;
using fcp::jni::TensorToBinaryProto;
using fcp::jni::ThrowCustomStatusCodeException;
using tensorflow::GraphDef;
using tensorflow::Session;
using tensorflow::SessionOptions;
using tensorflow::Status;
using tensorflow::Tensor;
using tensorflow::error::Code;

#define JFUN(METHOD_NAME) \
  Java_com_google_fcp_tensorflow_TensorflowSession_##METHOD_NAME

#define TF_EXCEPTION_CLASS "com/google/fcp/tensorflow/TensorflowException"

// Helper methods
// ==============
namespace {

// Throws a TensorflowException with the given status code and message in the
// JNI environment.
void ThrowTensorflowException(JNIEnv* env, int code,
                              const std::string& message) {
  ThrowCustomStatusCodeException(env, TF_EXCEPTION_CLASS, code, message);
}

ABSL_MUST_USE_RESULT
jlong createNativeCommon(JNIEnv* env, const GraphDef& graphDef) {
  // Create session
  SessionOptions options;
  options.config.mutable_graph_options()->set_place_pruned_graph(true);
  Session* session;
  Status s = tensorflow::NewSession(options, &session);
  if (!s.ok()) {
    ThrowTensorflowException(env, s.raw_code(), std::string(s.message()));
    return 0;
  }

  // Load graph
  s = session->Create(graphDef);
  if (!s.ok()) {
    ThrowTensorflowException(env, s.raw_code(), std::string(s.message()));
    return 0;
  }

  // Return session handle.
  return reinterpret_cast<jlong>(session);
}
}  // namespace

// JNI bindings
// ============

extern "C" JNIEXPORT jlong JNICALL JFUN(createNativeFromByteArray)(
    JNIEnv* env, jclass, jbyteArray graphDefByteArray) {
  absl::StatusOr<GraphDef> graphdef =
      ParseProtoFromJByteArray<GraphDef>(env, graphDefByteArray);
  if (!graphdef.ok()) {
    return 0;
  }

  return createNativeCommon(env, graphdef.value());
}

// Note: This method consumes (and destroys) the TensorFlow session.  After
// this method is called, the caller should never re-use the TensorFlow session.
extern "C" JNIEXPORT void JNICALL JFUN(closeNative)(JNIEnv* env, jobject obj,
                                                    jlong handle) {
  if (handle == 0) {
    ThrowTensorflowException(env, Code::INVALID_ARGUMENT,
                             "Invalid session handle (session closed?)");
    return;
  }
  std::unique_ptr<Session> session(reinterpret_cast<Session*>(handle));
  Status s = session->Close();
  if (!s.ok()) {
    ThrowTensorflowException(env, s.raw_code(), std::string(s.message()));
    return;
  }
}

extern "C" JNIEXPORT void JNICALL JFUN(runNative)(
    JNIEnv* env, jobject obj, jlong handle, jobjectArray inputTensorNames,
    jobjectArray inputTensorValues, jobjectArray outputTensorNames,
    jobjectArray outputTensorValues, jobjectArray target_nodeNames) {
  if (handle == 0) {
    ThrowTensorflowException(env, Code::INVALID_ARGUMENT,
                             "Invalid session handle (session closed?)");
    return;
  }
  Session* session = reinterpret_cast<Session*>(handle);

  std::vector<std::pair<std::string, Tensor>> inputs;
  int n = env->GetArrayLength(inputTensorNames);
  for (int i = 0; i < n; i++) {
    jstring jname = (jstring)(env->GetObjectArrayElement(inputTensorNames, i));
    JNI_FAILURE_CHECK(env, void());
    std::string name = JStringToString(env, jname);
    JNI_FAILURE_CHECK(env, void());
    jbyteArray jtensor =
        (jbyteArray)(env->GetObjectArrayElement(inputTensorValues, i));
    JNI_FAILURE_CHECK(env, void());
    absl::StatusOr<Tensor> tensor = BinaryProtoToTensor(env, jtensor);
    if (!tensor.ok()) {
      return;
    }
    inputs.emplace_back(name, tensor.value());
  }

  absl::StatusOr<std::vector<std::string>> output_names =
      JStringArrayToStringVector(env, outputTensorNames);
  if (!output_names.ok()) {
    return;
  }

  absl::StatusOr<std::vector<std::string>> target_nodes =
      JStringArrayToStringVector(env, target_nodeNames);
  if (!target_nodes.ok()) {
    return;
  }
  std::vector<Tensor> outputValues;

  Status s = session->Run(inputs, output_names.value(), target_nodes.value(),
                          &outputValues);
  if (!s.ok()) {
    ThrowTensorflowException(env, s.raw_code(), std::string(s.message()));
    return;
  }
  if (!outputValues.empty()) {
    for (int i = 0; i < outputValues.size(); i++) {
      absl::StatusOr<jbyteArray> binary_proto =
          TensorToBinaryProto(env, outputValues[i]);
      if (!binary_proto.ok()) {
        return;
      }
      env->SetObjectArrayElement(outputTensorValues, i, binary_proto.value());
      JNI_FAILURE_CHECK(env, void());
    }
  }
}

extern "C" JNIEXPORT void JNICALL JFUN(validateCheckpoint)(
    JNIEnv* env, jobject obj, jstring jfilepattern, jint max_tensor_size) {
  std::string filepattern = JStringToString(env, jfilepattern);
  JNI_FAILURE_CHECK(env, void());
  Status s = fcp::ValidateCheckpoint(filepattern, max_tensor_size);
  if (!s.ok()) {
    ThrowTensorflowException(env, s.raw_code(), std::string(s.message()));
    return;
  }
}
