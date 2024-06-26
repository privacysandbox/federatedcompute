# Copyright 2022 Google LLC
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

# Aggregation Protocol Package

load("@com_github_grpc_grpc//bazel:python_rules.bzl", "py_proto_library")
load("//fcp:config.bzl", "FCP_COPTS")

package(
    default_visibility = [
        "//fcp/aggregation:internal",
        "//fcp/client/engine:__pkg__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

proto_library(
    name = "proto",
    srcs = ["aggregation_protocol_messages.proto"],
    deps = [
        "//fcp/secagg/shared:proto",
    ],
)

cc_proto_library(
    name = "cc_proto",
    deps = [
        ":proto",
    ],
)

py_proto_library(
    name = "py_pb2",
    deps = [
        ":proto",
    ],
)

proto_library(
    name = "configuration_proto",
    srcs = ["configuration.proto"],
    deps = ["//fcp/aggregation/core:tensor_proto"],
)

cc_proto_library(
    name = "configuration_cc_proto",
    visibility = ["//visibility:public"],
    deps = [
        ":configuration_proto",
    ],
)

java_proto_library(
    name = "configuration_java_proto",
    visibility = ["//visibility:public"],
    deps = [
        ":configuration_proto",
    ],
)

py_proto_library(
    name = "configuration_py_pb2",
    deps = [
        ":configuration_proto",
    ],
)

cc_library(
    name = "aggregation_protocol",
    hdrs = [
        "aggregation_protocol.h",
    ],
    copts = FCP_COPTS,
    deps = [
        ":cc_proto",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:cord",
    ],
)

cc_library(
    name = "resource_resolver",
    hdrs = [
        "resource_resolver.h",
    ],
    copts = FCP_COPTS,
    deps = [
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:cord",
    ],
)

cc_library(
    name = "checkpoint_builder",
    hdrs = [
        "checkpoint_builder.h",
    ],
    copts = FCP_COPTS,
    deps = [
        "//fcp/aggregation/core:tensor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:cord",
    ],
)

cc_library(
    name = "checkpoint_parser",
    hdrs = [
        "checkpoint_parser.h",
    ],
    copts = FCP_COPTS,
    deps = [
        "//fcp/aggregation/core:tensor",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings:cord",
    ],
)

cc_library(
    name = "checkpoint_aggregator",
    srcs = ["checkpoint_aggregator.cc"],
    hdrs = ["checkpoint_aggregator.h"],
    copts = FCP_COPTS,
    visibility = ["//visibility:public"],
    deps = [
        ":checkpoint_builder",
        ":checkpoint_parser",
        ":config_converter",
        ":configuration_cc_proto",
        "//fcp/aggregation/core:aggregation_cores",
        "//fcp/aggregation/core:aggregator",
        "//fcp/aggregation/core:intrinsic",
        "//fcp/aggregation/core:tensor",
        "//fcp/base",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_test(
    name = "checkpoint_aggregator_test",
    srcs = ["checkpoint_aggregator_test.cc"],
    copts = FCP_COPTS,
    deps = [
        ":checkpoint_aggregator",
        ":configuration_cc_proto",
        "//fcp/aggregation/core:aggregator",
        "//fcp/aggregation/core:intrinsic",
        "//fcp/aggregation/core:tensor",
        "//fcp/aggregation/core:tensor_cc_proto",
        "//fcp/aggregation/testing",
        "//fcp/aggregation/testing:mocks",
        "//fcp/aggregation/testing:test_data",
        "//fcp/base",
        "//fcp/base:scheduler",
        "//fcp/testing",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/synchronization",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "config_converter",
    srcs = ["config_converter.cc"],
    hdrs = ["config_converter.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":configuration_cc_proto",
        "//fcp/aggregation/core:aggregator",
        "//fcp/aggregation/core:dp_fedsql_constants",
        "//fcp/aggregation/core:fedsql_constants",
        "//fcp/aggregation/core:intrinsic",
        "//fcp/aggregation/core:tensor",
        "//fcp/base",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_library(
    name = "federated_compute_checkpoint_builder",
    srcs = ["federated_compute_checkpoint_builder.cc"],
    hdrs = ["federated_compute_checkpoint_builder.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":checkpoint_builder",
        ":checkpoint_header",
        "//fcp/aggregation/core:tensor",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_test(
    name = "federated_compute_checkpoint_builder_test",
    srcs = ["federated_compute_checkpoint_builder_test.cc"],
    deps = [
        ":checkpoint_builder",
        ":checkpoint_header",
        ":federated_compute_checkpoint_builder",
        "//fcp/aggregation/core:tensor",
        "//fcp/aggregation/core:tensor_cc_proto",
        "//fcp/aggregation/testing:test_data",
        "//fcp/testing",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_googletest//:gtest_main",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_library(
    name = "federated_compute_checkpoint_parser",
    srcs = ["federated_compute_checkpoint_parser.cc"],
    hdrs = ["federated_compute_checkpoint_parser.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":checkpoint_header",
        ":checkpoint_parser",
        "//fcp/aggregation/core:tensor",
        "//fcp/aggregation/core:tensor_cc_proto",
        "//fcp/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_library(
    name = "checkpoint_header",
    hdrs = ["checkpoint_header.h"],
)

cc_test(
    name = "federated_compute_checkpoint_parser_test",
    srcs = ["federated_compute_checkpoint_parser_test.cc"],
    deps = [
        ":checkpoint_builder",
        ":checkpoint_parser",
        ":federated_compute_checkpoint_builder",
        ":federated_compute_checkpoint_parser",
        "//fcp/aggregation/core:tensor",
        "//fcp/aggregation/core:tensor_cc_proto",
        "//fcp/aggregation/testing",
        "//fcp/aggregation/testing:test_data",
        "//fcp/testing",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "config_converter_test",
    srcs = ["config_converter_test.cc"],
    deps = [
        ":config_converter",
        ":configuration_cc_proto",
        "//fcp/aggregation/core:aggregator",
        "//fcp/aggregation/core:intrinsic",
        "//fcp/aggregation/core:tensor",
        "//fcp/aggregation/core:tensor_cc_proto",
        "//fcp/aggregation/testing",
        "//fcp/aggregation/testing:test_data",
        "//fcp/base",
        "//fcp/testing",
        "//fcp/testing:parse_text_proto",
        "@com_google_googletest//:gtest_main",
    ],
)
