#include "fcp/aggregation/protocol/federated_compute_checkpoint_builder.h"

#include <cstdint>
#include <memory>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "fcp/aggregation/core/tensor.h"
#include "fcp/aggregation/core/tensor.pb.h"
#include "fcp/aggregation/core/tensor_shape.h"
#include "fcp/aggregation/protocol/checkpoint_builder.h"
#include "fcp/aggregation/testing/test_data.h"
#include "fcp/testing/testing.h"
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"

namespace fcp::aggregation {
namespace {

TEST(FederatedComputeCheckpointBuilderTest, BuildCheckpoint) {
  FederatedComputeCheckpointBuilderFactory factory;
  std::unique_ptr<CheckpointBuilder> builder = factory.Create();

  absl::StatusOr<Tensor> t1 = Tensor::Create(
      DT_INT64, TensorShape({3}), CreateTestData<uint64_t>({1, 2, 3}));
  ASSERT_OK(t1.status());
  absl::StatusOr<Tensor> t2 =
      Tensor::Create(DT_STRING, TensorShape({2}),
                     CreateTestData<absl::string_view>({"value1", "value2"}));
  ASSERT_OK(t2.status());

  EXPECT_OK(builder->Add("t1", *t1));
  EXPECT_OK(builder->Add("t2", *t2));
  absl::StatusOr<absl::Cord> checkpoint = builder->Build();
  ASSERT_OK(checkpoint.status());

  absl::string_view str = checkpoint->Flatten();
  google::protobuf::io::ArrayInputStream input(str.data(), static_cast<int>(str.size()));
  google::protobuf::io::CodedInputStream stream(&input);

  std::string header;
  ASSERT_TRUE(stream.ReadString(&header, 4));
  ASSERT_EQ(header, kFederatedComputeCheckpointHeader);

  uint32_t name_size1;
  ASSERT_TRUE(stream.ReadVarint32(&name_size1));
  std::string name1;
  ASSERT_TRUE(stream.ReadString(&name1, name_size1));
  ASSERT_EQ(name1, "t1");
  uint32_t tensor_size1;
  ASSERT_TRUE(stream.ReadVarint32(&tensor_size1));
  std::string tensor1;
  ASSERT_TRUE(stream.ReadString(&tensor1, tensor_size1));
  ASSERT_EQ(tensor1, (*t1).ToProto().SerializeAsString());

  uint32_t name_size2;
  ASSERT_TRUE(stream.ReadVarint32(&name_size2));
  std::string name2;
  ASSERT_TRUE(stream.ReadString(&name2, name_size2));
  ASSERT_EQ(name2, "t2");
  uint32_t tensor_size2;
  ASSERT_TRUE(stream.ReadVarint32(&tensor_size2));
  std::string tensor2;
  ASSERT_TRUE(stream.ReadString(&tensor2, tensor_size2));
  ASSERT_EQ(tensor2, (*t2).ToProto().SerializeAsString());

  uint32_t zero;
  ASSERT_TRUE(stream.ReadVarint32(&zero));
  ASSERT_EQ(zero, 0);
}
}  // namespace
}  // namespace fcp::aggregation
