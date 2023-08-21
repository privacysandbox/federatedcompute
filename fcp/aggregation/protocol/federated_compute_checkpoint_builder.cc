#include "fcp/aggregation/protocol/federated_compute_checkpoint_builder.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "fcp/aggregation/core/tensor.h"
#include "fcp/aggregation/protocol/checkpoint_builder.h"
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"

namespace fcp::aggregation {

namespace {
// Builds and formats a set of aggregation tensors using the new wire format for
// federated compute.
class FederatedComputeCheckpointBuilder final : public CheckpointBuilder {
 public:
  FederatedComputeCheckpointBuilder() {
    // Indicates that the checkpoint is using the new wire format.
    result_.Append(kFederatedComputeCheckpointHeader);
  }
  // Disallow copy and move constructors.
  FederatedComputeCheckpointBuilder(const FederatedComputeCheckpointBuilder&) =
      delete;
  FederatedComputeCheckpointBuilder& operator=(
      const FederatedComputeCheckpointBuilder&) = delete;

  absl::Status Add(const std::string& name, const Tensor& tensor) override {
    std::string metadata;
    google::protobuf::io::StringOutputStream out(&metadata);
    google::protobuf::io::CodedOutputStream coded_out(&out);
    coded_out.WriteVarint64(name.size());
    coded_out.WriteString(name);

    absl::Cord content(tensor.ToProto().SerializeAsString());
    if (content.empty()) {
      return absl::InternalError("Failed to add tensor for " + name);
    }
    coded_out.WriteVarint64(content.size());
    coded_out.Trim();
    result_.Append(metadata);
    result_.Append(content);
    return absl::OkStatus();
  }

  absl::StatusOr<absl::Cord> Build() override {
    uint32_t zero = 0;
    result_.Append(
        absl::string_view(reinterpret_cast<const char*>(&zero), sizeof(zero)));
    return result_;
  }

 private:
  absl::Cord result_;
};

}  // namespace

std::unique_ptr<CheckpointBuilder>
FederatedComputeCheckpointBuilderFactory::Create() const {
  return std::make_unique<FederatedComputeCheckpointBuilder>();
}

}  // namespace fcp::aggregation
