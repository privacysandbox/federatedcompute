# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
load("@rules_java//java:defs.bzl", "java_proto_library")

package(
    default_visibility = ["//visibility:public"],
)

java_library(
    name = "tensorflow",
    srcs = [
        "TensorflowException.java",
        "TensorflowSession.java",
    ],
    runtime_deps = [
        "//fcp/java_src/main/java/com/google/fcp/tensorflow/jni:libtensorflow-jni.so",
    ],
    deps = [
        ":app_files",
        ":file_util",
        ":native_handle",
        ":protos_all_java_proto",
        "//fcp/protos:plan_java_proto",
        "@com_google_protobuf//java/core",
        "@maven//:com_google_code_findbugs_jsr305",
        "@maven//:com_google_guava_guava",
    ],
)

java_library(
    name = "native_handle",
    srcs = ["NativeHandle.java"],
    deps = ["@maven//:com_google_guava_guava"],
)

java_library(
    name = "app_files",
    srcs = ["AppFiles.java"],
    deps = [],
)

java_library(
    name = "file_util",
    srcs = ["FileUtil.java"],
    deps = [
        "@com_google_protobuf//java/core",
    ],
)

java_proto_library(
    name = "protos_all_java_proto",
    deps = [
        "@org_tensorflow//tensorflow/core:protos_all",
    ],
)

java_library(
    name = "tensorflow_exception",
    srcs = ["TensorflowException.java"],
    deps = [],
)
