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

#ifndef FCP_JNI_MORE_JNI_UTIL_H_
#define FCP_JNI_MORE_JNI_UTIL_H_

#include <fcntl.h>
#include <jni.h>
#include <string.h>

#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "fcp/base/monitoring.h"
#include "fcp/jni/jni_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"

using fcp::jni::ParseProtoFromJByteArray;
using fcp::jni::SerializeProtoToJByteArray;

// Checks whether a JNI call failed. This is meant to be called right after any
// JNI call, to detect error conditions as early as possible. For functions that
// need to return "void" on failure you can specify "void()" as the return_val.
#define JNI_FAILURE_CHECK(env, return_val)              \
  while (ABSL_PREDICT_FALSE((env)->ExceptionCheck())) { \
    env->ExceptionClear();                              \
    return return_val;                                  \
  }

namespace fcp {
namespace jni {

static std::string JStringToString(JNIEnv* env, jstring jstr) {
  if (jstr == nullptr) {
    return std::string();
  }
  const char* cstring = env->GetStringUTFChars(jstr, nullptr);
  FCP_CHECK(!env->ExceptionCheck());
  std::string result(cstring);
  env->ReleaseStringUTFChars(jstr, cstring);
  return result;
}

// Converts a Java array of String objects into a vector of C++ strings. If an
// error is returned then an exception will already have been thrown in the
// JNI env and callers should return to Java ASAP.
static absl::StatusOr<std::vector<std::string>> JStringArrayToStringVector(
    JNIEnv* env, jobjectArray stringArray) {
  std::vector<std::string> result;
  jsize n = env->GetArrayLength(stringArray);
  FCP_CHECK(!env->ExceptionCheck());
  result.reserve(n);
  for (jsize i = 0; i < n; i++) {
    jstring inner_jstring = (jstring)env->GetObjectArrayElement(stringArray, i);
    FCP_CHECK(!env->ExceptionCheck());
    result.push_back(JStringToString(env, inner_jstring));
    FCP_CHECK(!env->ExceptionCheck());
  }
  // Ensure StatusOr's move constructor is used.
  return std::move(result);
}

// Converts the given Tensor into TensorProto, and then serializes that proto
// into Java byte array. If an error is returned then an exception will
// already have been thrown in the JNI env and callers should return to Java
// ASAP.
static absl::StatusOr<jbyteArray> TensorToBinaryProto(
    JNIEnv* env, const tensorflow::Tensor& tensor) {
  tensorflow::TensorProto proto;
  // Populate field view so Java can see the content.
  tensor.AsProtoField(&proto);
  return SerializeProtoToJByteArray(env, proto);
}

// Parses a TensorProto from a Java byte array, and loads it into a Tensor.
//
// If the parsing of the proto fails, then this returns an INVALID_ARGUMENT
// error. If a JNI error occurs then this returns an INTERNAL error, and a
// runtime exception will be thrown. In both cases the caller should return to
// Java ASAP.
static absl::StatusOr<tensorflow::Tensor> BinaryProtoToTensor(
    JNIEnv* env, jbyteArray byteArray) {
  absl::StatusOr<tensorflow::TensorProto> proto =
      ParseProtoFromJByteArray<tensorflow::TensorProto>(env, byteArray);
  if (!proto.ok()) {
    return proto.status();
  }
  tensorflow::Tensor tensor;
  if (!tensor.FromProto(proto.value())) {
    return absl::Status(absl::StatusCode::kInvalidArgument,
                        "Cannot convert tensor after deserialization");
  }
  // Ensure StatusOr's move constructor is used.
  return std::move(tensor);
}
}  // namespace jni
}  // namespace fcp

#endif  // FCP_JNI_MORE_JNI_UTIL_H_