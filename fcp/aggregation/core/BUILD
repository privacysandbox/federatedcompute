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

# Aggregation Core API

load("@com_github_grpc_grpc//bazel:python_rules.bzl", "py_proto_library")
load("//fcp:config.bzl", "FCP_COPTS")

package(
    default_visibility = [
        "//fcp/aggregation:internal",
        "//fcp/client:__pkg__",
        "//fcp/client/engine:__pkg__",
    ],
    licenses = ["notice"],  # Apache 2.0
)

# TODO(team): Create a "core" library that bundles all core libraries together.

proto_library(
    name = "tensor_proto",
    srcs = ["tensor.proto"],
)

cc_proto_library(
    name = "tensor_cc_proto",
    deps = [":tensor_proto"],
)

java_proto_library(
    name = "tensor_java_proto",
    visibility = ["//visibility:public"],
    deps = [":tensor_proto"],
)

py_proto_library(
    name = "tensor_py_pb2",
    visibility = ["//visibility:public"],
    deps = [
        ":tensor_proto",
    ],
)

proto_library(
    name = "agg_core_proto",
    srcs = ["agg_core.proto"],
    deps = [":tensor_proto"],
)

cc_proto_library(
    name = "agg_core_cc_proto",
    deps = [":agg_core_proto"],
)

cc_library(
    name = "tensor",
    srcs = [
        "datatype.cc",
        "input_tensor_list.cc",
        "tensor.cc",
        "tensor_data.cc",
        "tensor_shape.cc",
        "tensor_spec.cc",
    ],
    hdrs = [
        "agg_vector.h",
        "agg_vector_iterator.h",
        "datatype.h",
        "input_tensor_list.h",
        "mutable_vector_data.h",
        "tensor.h",
        "tensor_data.h",
        "tensor_shape.h",
        "tensor_spec.h",
    ],
    copts = FCP_COPTS,
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":tensor_cc_proto",
        "//fcp/base",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_protobuf//:protobuf",
    ],
)

cc_library(
    name = "aggregator",
    srcs = [
        "tensor_aggregator.cc",
        "tensor_aggregator_registry.cc",
    ],
    hdrs = [
        "agg_vector_aggregator.h",
        "aggregator.h",
        "one_dim_grouping_aggregator.h",
        "tensor_aggregator.h",
        "tensor_aggregator_factory.h",
        "tensor_aggregator_registry.h",
    ],
    copts = FCP_COPTS,
    deps = [
        ":agg_core_cc_proto",
        ":intrinsic",
        ":tensor",
        ":tensor_cc_proto",
        "//fcp/base",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "intrinsic",
    hdrs = [
        "intrinsic.h",
    ],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":tensor",
    ],
)

cc_library(
    name = "aggregation_cores",
    srcs = [
        "composite_key_combiner.cc",
        "dp_composite_key_combiner.cc",
        "dp_group_by_aggregator.cc",
        "dp_grouping_federated_sum.cc",
        "federated_mean.cc",
        "federated_sum.cc",
        "group_by_aggregator.cc",
        "grouping_federated_sum.cc",
        "one_dim_grouping_aggregator.cc",
    ],
    hdrs = [
        "composite_key_combiner.h",
        "dp_composite_key_combiner.h",
        "dp_group_by_aggregator.h",
        "group_by_aggregator.h",
        "one_dim_grouping_aggregator.h",
    ],
    copts = FCP_COPTS,
    deps = [
        ":agg_core_cc_proto",
        ":aggregator",
        ":dp_fedsql_constants",
        ":fedsql_constants",
        ":intrinsic",
        ":tensor",
        ":tensor_cc_proto",
        ":vector_string_data",
        "//fcp/base",
        "@com_google_absl//absl/container:fixed_array",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/container:flat_hash_set",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/random",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:span",
        "@com_google_cc_differential_privacy//algorithms:numerical-mechanisms",
        "@com_google_cc_differential_privacy//algorithms:partition-selection",
        "@com_google_protobuf//:protobuf",
    ],
    alwayslink = 1,
)

cc_test(
    name = "tensor_test",
    srcs = [
        "tensor_data_test.cc",
        "tensor_shape_test.cc",
        "tensor_spec_test.cc",
        "tensor_test.cc",
    ],
    copts = FCP_COPTS,
    deps = [
        ":tensor",
        ":tensor_cc_proto",
        "//fcp/aggregation/testing",
        "//fcp/aggregation/testing:test_data",
        "//fcp/base",
        "//fcp/testing",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "agg_vector_test",
    srcs = ["agg_vector_test.cc"],
    copts = FCP_COPTS,
    deps = [
        ":tensor",
        "//fcp/aggregation/testing:test_data",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "aggregator_test",
    srcs = [
        "agg_vector_aggregator_test.cc",
    ],
    copts = FCP_COPTS,
    deps = [
        ":agg_core_cc_proto",
        ":aggregator",
        ":tensor",
        ":tensor_cc_proto",
        "//fcp/aggregation/testing",
        "//fcp/aggregation/testing:test_data",
        "//fcp/base",
        "//fcp/testing",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "tensor_aggregator_registry_test",
    srcs = [
        "tensor_aggregator_registry_test.cc",
    ],
    copts = FCP_COPTS,
    deps = [
        ":aggregator",
        ":intrinsic",
        "//fcp/base",
        "//fcp/testing",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_library(
    name = "vector_string_data",
    hdrs = ["vector_string_data.h"],
    deps = [":tensor"],
)

cc_library(
    name = "mutable_string_data",
    hdrs = ["mutable_string_data.h"],
    visibility = [
        "//visibility:public",
    ],
    deps = [
        ":tensor",
        "@com_google_absl//absl/container:fixed_array",
    ],
)

cc_library(
    name = "fedsql_constants",
    hdrs = ["fedsql_constants.h"],
)

cc_library(
    name = "dp_fedsql_constants",
    hdrs = ["dp_fedsql_constants.h"],
)

cc_test(
    name = "federated_mean_test",
    srcs = ["federated_mean_test.cc"],
    copts = FCP_COPTS,
    deps = [
        ":agg_core_cc_proto",
        ":aggregation_cores",
        ":aggregator",
        ":intrinsic",
        ":tensor",
        "//fcp/aggregation/testing",
        "//fcp/aggregation/testing:test_data",
        "//fcp/base",
        "//fcp/testing",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "federated_sum_test",
    srcs = ["federated_sum_test.cc"],
    copts = FCP_COPTS,
    deps = [
        ":agg_core_cc_proto",
        ":aggregation_cores",
        ":aggregator",
        ":tensor",
        "//fcp/aggregation/testing",
        "//fcp/aggregation/testing:test_data",
        "//fcp/testing",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "input_tensor_list_test",
    srcs = ["input_tensor_list_test.cc"],
    deps = [
        ":tensor",
        "//fcp/aggregation/testing:test_data",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "composite_key_combiner_test",
    srcs = ["composite_key_combiner_test.cc"],
    deps = [
        ":aggregation_cores",
        ":aggregator",
        ":tensor",
        ":tensor_cc_proto",
        "//fcp/aggregation/testing",
        "//fcp/aggregation/testing:test_data",
        "//fcp/base",
        "//fcp/testing",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "dp_composite_key_combiner_test",
    srcs = ["dp_composite_key_combiner_test.cc"],
    deps = [
        ":aggregation_cores",
        ":aggregator",
        ":tensor",
        ":tensor_cc_proto",
        "//fcp/aggregation/testing",
        "//fcp/aggregation/testing:test_data",
        "//fcp/base",
        "//fcp/testing",
        "@com_google_absl//absl/types:span",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "mutable_vector_data_test",
    srcs = ["mutable_vector_data_test.cc"],
    deps = [
        ":tensor",
        "//fcp/testing",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "vector_string_data_test",
    srcs = ["vector_string_data_test.cc"],
    deps = [
        ":tensor",
        ":vector_string_data",
        "//fcp/testing",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "mutable_string_data_test",
    srcs = ["mutable_string_data_test.cc"],
    deps = [
        ":mutable_string_data",
        ":tensor",
        "//fcp/testing",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "one_dim_grouping_aggregator_test",
    srcs = ["one_dim_grouping_aggregator_test.cc"],
    deps = [
        ":agg_core_cc_proto",
        ":aggregation_cores",
        ":tensor",
        "//fcp/aggregation/testing",
        "//fcp/aggregation/testing:test_data",
        "//fcp/base",
        "//fcp/testing",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "group_by_aggregator_test",
    srcs = ["group_by_aggregator_test.cc"],
    deps = [
        ":aggregation_cores",
        ":aggregator",
        ":intrinsic",
        ":tensor",
        ":tensor_cc_proto",
        "//fcp/aggregation/testing",
        "//fcp/aggregation/testing:test_data",
        "//fcp/base",
        "//fcp/testing",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "dp_group_by_aggregator_test",
    srcs = ["dp_group_by_aggregator_test.cc"],
    deps = [
        ":aggregation_cores",
        ":aggregator",
        ":dp_fedsql_constants",
        ":intrinsic",
        ":tensor",
        ":tensor_cc_proto",
        "//fcp/aggregation/testing",
        "//fcp/aggregation/testing:test_data",
        "//fcp/base",
        "//fcp/testing",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "grouping_federated_sum_test",
    srcs = ["grouping_federated_sum_test.cc"],
    copts = FCP_COPTS,
    deps = [
        ":aggregation_cores",
        ":aggregator",
        ":intrinsic",
        ":tensor",
        "//fcp/aggregation/testing",
        "//fcp/aggregation/testing:test_data",
        "//fcp/base",
        "//fcp/testing",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "dp_grouping_federated_sum_test",
    srcs = ["dp_grouping_federated_sum_test.cc"],
    copts = FCP_COPTS,
    deps = [
        ":aggregation_cores",
        ":aggregator",
        ":dp_fedsql_constants",
        ":intrinsic",
        ":tensor",
        ":tensor_cc_proto",
        "//fcp/aggregation/testing",
        "//fcp/aggregation/testing:test_data",
        "//fcp/base",
        "//fcp/testing",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_binary(
    name = "federated_sum_bench",
    testonly = 1,
    srcs = ["federated_sum_bench.cc"],
    copts = FCP_COPTS,
    linkstatic = 1,
    deps = [
        ":aggregation_cores",
        ":aggregator",
        ":intrinsic",
        ":tensor",
        "@com_google_benchmark//:benchmark_main",
    ],
)

cc_binary(
    name = "group_by_bench",
    testonly = 1,
    srcs = ["group_by_bench.cc"],
    copts = FCP_COPTS,
    linkstatic = 1,
    deps = [
        ":aggregation_cores",
        ":aggregator",
        ":intrinsic",
        ":mutable_string_data",
        ":tensor",
        ":tensor_cc_proto",
        "@com_google_benchmark//:benchmark_main",
    ],
)
