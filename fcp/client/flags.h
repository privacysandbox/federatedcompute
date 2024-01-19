/*
 * Copyright 2020 Google LLC
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
#ifndef FCP_CLIENT_FLAGS_H_
#define FCP_CLIENT_FLAGS_H_

#include <cstdint>

#include "absl/status/status.h"

namespace fcp {
namespace client {

// A class for changing runtime behavior with "flags" - typically, server
// provided values.
class Flags {
 public:
  virtual ~Flags() = default;

  // The period of time in milliseconds between device condition checks. This is
  // used during potentially long blocking calls such as TensorFlow or network
  // I/O, as well as for throttling regular condition checks during plan
  // execution (e.g. before fetching a new example).
  virtual int64_t condition_polling_period_millis() const = 0;

  // The period of time in milliseconds allowed for TensorFlow execution to
  // finish after it's been interrupted.
  virtual int64_t tf_execution_teardown_grace_period_millis() const = 0;

  // The period of time in milliseconds allowed for TensorFlow execution to
  // finish after the grace period. This allows us to decide if we want long
  // running native execution to be forcibly resolved or continue indefinitely.
  virtual int64_t tf_execution_teardown_extended_period_millis() const = 0;

  // The deadline in seconds for the gRPC channel used for communication
  // between the client and server.
  virtual int64_t grpc_channel_deadline_seconds() const = 0;

  // Whether to log the error message strings from TensorFlow exceptions.
  virtual bool log_tensorflow_error_messages() const = 0;

  // The number of days for data to live in the OpStatsDb without update.
  virtual int64_t opstats_ttl_days() const { return 30; }

  // The maximum size of the data stored by OpStatsDb.
  virtual int64_t opstats_db_size_limit_bytes() const {
    return 1 * 1024 * 1024;
  }

  // The retry delay to use when encountering a transient error during a
  // training run before having received a RetryWindow from the server.
  virtual int64_t federated_training_transient_errors_retry_delay_secs() const {
    // 15 minutes
    return 15 * 60;
  }

  // The amount of jitter to apply when using the
  // `federated_training_transient_errors_retry_delay_secs` flag. Must be a
  // value between 0 and 1. E.g. a value of 0.2 means that retry delays will
  // fall within [0.8 * target period, 1.2 * target period).
  virtual float federated_training_transient_errors_retry_delay_jitter_percent()
      const {
    return 0.2;
  }

  // The retry delay to use when encountering a permanent error during a
  // training run (regardless of whether the client already received a
  // RetryWindow from the server).
  virtual int64_t federated_training_permanent_errors_retry_delay_secs() const {
    // 4 hours
    return 4 * 60 * 60;
  }

  // The amount of jitter to apply when using the
  // `federated_training_permanent_errors_retry_delay_secs` flag. Must be a
  // value between 0 and 1. E.g. a value of 0.2 means that retry delays will
  // fall within [0.8 * target period, 1.2 * target period).
  virtual float federated_training_permanent_errors_retry_delay_jitter_percent()
      const {
    return 0.2;
  }

  // The list of error codes that should be considered 'permanent'.
  virtual std::vector<int32_t> federated_training_permanent_error_codes()
      const {
    return {
        // The server returns NOT_FOUND if the client checks in with an unknown
        // population name. While this can be resolved without any client
        // changes by creating the population server-side, it is nevertheless
        // wise to treat this as a 'permanent' error for which a longer
        // RetryPeriod is used, because such temporary mismatches in
        // client/server configuration are fairly common and otherwise cause
        // clients to check in unnecessarily frequently.
        static_cast<int32_t>(absl::StatusCode::kNotFound),
        // INVALID_ARGUMENT generally indicates a client-side issue (e.g. a bug
        // in the client's protocol implementation), which is unlikely to be
        // resolved by merely retrying the request.
        static_cast<int32_t>(absl::StatusCode::kInvalidArgument),
        // UNIMPLEMENTED similarly could indicate a client-side issue, or a
        // temporary server issue (e.g. a bug/missing feature implementation in
        // the server). Either way, it is also unlikely to be resolved by merely
        // retrying the request soon.
        static_cast<int32_t>(absl::StatusCode::kUnimplemented)};
  }

  // Whether use TFLite for training.
  virtual bool use_tflite_training() const { return false; }

  // Whether to enable support for downloading plan/initial checkpoint resources
  // via HTTP, while still using gRPC for the main protocol.
  virtual bool enable_grpc_with_http_resource_support() const { return false; }

  // Whether to enable support for downloading eligibility eval plan/initial
  // checkpoint resources via HTTP, while still using gRPC for the main
  // protocol.
  virtual bool enable_grpc_with_eligibility_eval_http_resource_support() const {
    return false;
  }

  // When true, TFLite interpreter will use dynamic memory allocation, and
  // release the memory for tensors that are no longer needed.
  virtual bool ensure_dynamic_tensors_are_released() const { return true; }

  // When the value is above zero, any tensor size (bytes) above the threshold
  // will be considered as a large tensor, and dynamic allocation is applied on
  // them.
  virtual int32_t large_tensor_threshold_for_dynamic_allocation() const {
    return 1000;
  }

  // When true, the TFLite runtime graph-reordering optimization that clusters
  // delegate nodes together is disabled.
  virtual bool disable_tflite_delegate_clustering() const { return false; }

  // When true, http request body won't be compressed.
  virtual bool disable_http_request_body_compression() const { return false; }

  // When true, HTTP Federated Compute protocol is used.
  virtual bool use_http_federated_compute_protocol() const { return false; }

  // When true, the client computes the task identity to pass in
  // SelectorContext.
  virtual bool enable_computation_id() const { return false; }

  // The waiting period for issuing cancellation requests before checking
  // whether the client should be interrupted.
  virtual int32_t waiting_period_sec_for_cancellation() const { return 10; }

  // If true, the client supports the Federated Select feature. If not
  // then any Federated Select-specific example query will fail with an error
  virtual bool enable_federated_select() const { return false; }

  // The max size in bytes of resources that the ResourceCache is allowed to
  // store. If greater than 0, the client will attempt to cache resources sent
  // by uri via the hybrid grpc-with-http-resources and the full http stack. If
  // this value is reduced from some previous greater value, the cache dir will
  // be reduced appropriately the next time it is initialized at the start of
  // the next run.
  virtual int64_t max_resource_cache_size_bytes() const { return 0; }

  // If true, an error during the initialization of the resource cache will
  // publish a fatal initialization error instead of a nonfatal initialization
  // error and halt execution.
  virtual bool resource_cache_initialization_error_is_fatal() const {
    return false;
  }

  // The number of threads that TFLite interpreter will use.
  virtual int32_t num_threads_for_tflite() const { return 2; }

  // If true, enables an Example Query plan engine to be invoked for
  // non-TensorFlow tasks.
  virtual bool enable_example_query_plan_engine() const { return false; }

  // If true, the HTTP federated protocol supports multiple task assignments.
  virtual bool http_protocol_supports_multiple_task_assignments() const {
    return false;
  }

  // If true, enables support for native computation of Eligibility Eval Tasks.
  virtual bool enable_native_eets() const { return false; }

  // If true, enables support for TensorFlow custom policies in the native
  // Eligibility Eval stack.
  virtual bool neet_tf_custom_policy_support() const { return false; }

  // If true, failing to evaluate an eligibility policy will opt the device out
  // of tasks using that policy instead of halting execution. Requires
  // neet_tf_custom_policy_support.
  virtual bool graceful_eligibility_policy_failure() const { return false; }

  // If true, enables new client report wire format for lightweight client.
  virtual bool enable_lightweight_client_report_wire_format() const {
    return false;
  }

  // If true, OpStats logger enables PhaseStats logging.
  virtual bool enable_phase_stats_logging() const { return false; }

  // If true, log the first access time for each collection to opstats.
  virtual bool log_collection_first_access_time() const { return false; }

  // If true, issue single task assignment only when the
  // EligibilityPopulationSpec indicates the population supports single task
  // assignment, or when the EligibilityPopulationSpec is missing.
  virtual bool check_eligibility_population_spec_before_checkin() const {
    return false;
  }

  // If true, a callback will be called when a task completes.  For multiple
  // task assignments, the callback could be called multiple times.
  virtual bool enable_task_completion_callback() const { return false; }

  // If true, example query recording will happen in the native code. But if
  // enable_task_completion_callback is false, the recorded data won't be used.
  virtual bool enable_native_example_query_recording() const { return false; }

  // If true, confidential aggregation is advertised & can be used for uploads.
  virtual bool enable_confidential_aggregation() const { return false; }

  // If true, enables MinimumSeparationPolicy for clients.
  virtual bool enable_minimum_separation_policy() const { return false; }
};
}  // namespace client
}  // namespace fcp

#endif  // FCP_CLIENT_FLAGS_H_
