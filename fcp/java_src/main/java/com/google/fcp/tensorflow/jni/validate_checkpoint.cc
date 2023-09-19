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

#include "validate_checkpoint.h"

#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/resource_handle.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/util/tensor_slice_reader.h"

namespace fcp {
namespace {

// Returns the size of an empty data type. Note that this under-estimates the
// size of complex data types like tstring, but it still provides a rough
// estimate that should be good enough to prevent massive memory allocations.
tensorflow::StatusOr<int> GetDataTypeSize(tensorflow::DataType data_type) {
  // tensorflow::DataTypeSize doesn't support several types.
  switch (data_type) {
    case tensorflow::DT_STRING:
      return sizeof(tensorflow::tstring);
    case tensorflow::DT_VARIANT:
      return sizeof(tensorflow::Variant);
    case tensorflow::DT_RESOURCE:
      return sizeof(tensorflow::ResourceHandle);
    default: {
      int size = tensorflow::DataTypeSize(data_type);
      if (size == 0) {
        return tensorflow::Status(
            absl::StatusCode::kInternal,
            absl::StrCat("Unable to determine DataTypeSize for ",
                         tensorflow::DataType_Name(data_type)));
      }
      return size;
    }
  }
}

}  // namespace

tensorflow::Status ValidateCheckpoint(const std::string& filepattern,
                                      int max_tensor_size) {
  tensorflow::checkpoint::TensorSliceReader reader(filepattern);
  TF_RETURN_IF_ERROR(reader.status());

  for (const auto& [name, tss] : reader.Tensors()) {
    TF_ASSIGN_OR_RETURN(int type_size, GetDataTypeSize(tss->type()));
    if (tss->shape().num_elements() > max_tensor_size / type_size) {
      return tensorflow::Status(
          absl::StatusCode::kInvalidArgument,
          absl::StrCat(
              "Size of tensor ", name, " (", tss->shape().num_elements(),
              " elements * ", type_size,
              " bytes per element) exceeds the maximum permitted size (",
              max_tensor_size, ")"));
    }
  }
  return ::tensorflow::OkStatus();
}

}  // namespace fcp