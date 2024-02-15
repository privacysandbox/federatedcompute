/*
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef FCP_AGGREGATION_PROTOCOL_FEDERATED_COMPUTE_CHECKPOINT_PARSER_H_
#define FCP_AGGREGATION_PROTOCOL_FEDERATED_COMPUTE_CHECKPOINT_PARSER_H_

#include <memory>

#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "fcp/aggregation/protocol/checkpoint_parser.h"

namespace fcp::aggregation {

// A CheckpointParserFactory implementation that creates federated compute wire
// format checkpoint parser.
class FederatedComputeCheckpointParserFactory : public CheckpointParserFactory {
 public:
  absl::StatusOr<std::unique_ptr<CheckpointParser>> Create(
      const absl::Cord& serialized_checkpoint) const override;
};

}  // namespace fcp::aggregation

#endif  // FCP_AGGREGATION_PROTOCOL_FEDERATED_COMPUTE_CHECKPOINT_PARSER_H_
