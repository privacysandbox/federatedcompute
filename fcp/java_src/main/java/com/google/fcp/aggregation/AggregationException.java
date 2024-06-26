// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.fcp.aggregation;

/** Exception thrown when interaction with aggregation fails. */
public final class AggregationException extends Exception {

  // Equivalent to org.tensorflow.framework.Code.UNKNOWN
  public static final int UNKNOWN = 2;
  private final int errorCode;

  public AggregationException(String message) {
    super(message);
    this.errorCode = UNKNOWN;
  }

  public AggregationException(String message, Throwable t) {
    super(message, t);
    this.errorCode = UNKNOWN;
  }

  public AggregationException(int errorCode, String message) {
    super(message);
    this.errorCode = errorCode;
  }

  public AggregationException(int errorCode, String message, Throwable t) {
    super(message, t);
    this.errorCode = errorCode;
  }

  public int getErrorCode() {
    return errorCode;
  }
}
