/*
 * Copyright 2022 Google LLC
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

#ifndef FCP_AGGREGATION_CORE_TENSOR_SHAPE_H_
#define FCP_AGGREGATION_CORE_TENSOR_SHAPE_H_

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <utility>
#include <vector>

#include "fcp/base/monitoring.h"

#ifndef FCP_NANOLIBC
#include "fcp/aggregation/core/tensor.pb.h"
#endif

namespace fcp {
namespace aggregation {

// Represents a tensor shape as a collection of
// dimension sizes.
class TensorShape final {
 public:
  using DimSizesVector = std::vector<int64_t>;

  template <typename ForwardIterator>
  TensorShape(ForwardIterator first, ForwardIterator last)
      : dim_sizes_(first, last) {
    Status status = CheckValidDimSizes(dim_sizes_);
    FCP_CHECK(status.ok()) << status.message();
  }

  TensorShape(std::initializer_list<int64_t> dim_sizes)
      : TensorShape(dim_sizes.begin(), dim_sizes.end()) {}

#ifndef FCP_NANOLIBC
  // Creates a TensorShape from a TensorShapeProto.
  // Returns an error if any of the shape dimensions are unknown.
  static StatusOr<TensorShape> FromProto(const TensorShapeProto& shape_proto);

  // Returns a TensorShapeProto representation of the tensor shape.
  TensorShapeProto ToProto() const;
#endif

  // Gets the dimensions and their sizes.
  const DimSizesVector& dim_sizes() const { return dim_sizes_; }

  // Gets the total number of elements (which is a multiplication of sizes of
  // all dimensions).
  // For a scalar tensor with zero dimensions this returns 1.
  // For a tensor with any unknown dimensions this returns an INVALID_ARGUMENT
  // status.
  StatusOr<size_t> NumElements() const;

  // Returns true if the dimensions of known size in this TensorShape match the
  // sizes of corresponding dimensions in `other`. `other` is only permitted to
  // have dimensions of unknown size at the same dimensions as `this`.
  bool MatchesKnownDimensions(const TensorShape& other) const;

  friend bool operator==(const TensorShape& a, const TensorShape& b) {
    return a.dim_sizes_ == b.dim_sizes_;
  }

  friend bool operator!=(const TensorShape& a, const TensorShape& b) {
    return a.dim_sizes_ != b.dim_sizes_;
  }

 private:
  explicit TensorShape(DimSizesVector&& dim_sizes)
      : dim_sizes_(std::move(dim_sizes)) {}

  static Status CheckValidDimSizes(const DimSizesVector& dim_sizes) {
    for (auto dim_size : dim_sizes) {
      if (dim_size < -1) {
        return FCP_STATUS(INVALID_ARGUMENT)
               << "TensorShape: Dimension size less than -1 isn't supported.";
      }
    }
    return FCP_STATUS(OK);
  }

  // TODO(team): Consider optimizing the storage for better inlining
  // of small number of dimensions.
  DimSizesVector dim_sizes_;
};

}  // namespace aggregation
}  // namespace fcp

#endif  // FCP_AGGREGATION_CORE_TENSOR_SHAPE_H_
