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

package(
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "native",
    srcs = [
        "tensorflow_jni.cc",
        "validate_checkpoint.cc",
    ],
    hdrs = [
        "validate_checkpoint.h",
    ],
    copts = [
        "-Wno-narrowing",
        "-Wno-sign-compare",
        "-Wno-overloaded-virtual",
    ],
    deps = [
        "//fcp/jni:jni_util",
        "//fcp/jni:more_jni_util",
        "@bazel_tools//tools/jdk:jni",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/status:statusor",
        "@org_tensorflow//tensorflow/core:core_cpu",
        "@org_tensorflow//tensorflow/core:framework",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
        "@org_tensorflow//tensorflow/core:tensorflow",
    ],
    alwayslink = 1,
)

cc_binary(
    name = "libtensorflow-jni.so",
    linkshared = 1,
    linkstatic = 1,
    deps = [":native"],
)
