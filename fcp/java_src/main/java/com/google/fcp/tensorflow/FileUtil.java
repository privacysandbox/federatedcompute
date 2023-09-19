// Copyright 2023 Google LLC
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
package com.google.fcp.tensorflow;

import static java.lang.Math.min;

import com.google.protobuf.ByteString;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;

/**
 * Utilities for writing and reading binary data to/from a file.
 */
final class FileUtil {
  private static final int CHUNK_SIZE = 0x2000; // 8k

  /**
   * Reads data from an {@link InputByteStream} as a {@link ByteString}.
   */
  static ByteString readByteString(FileInputStream stream) throws IOException {
    return ByteString.readFrom(stream, CHUNK_SIZE);
  }

  /**
   * Writes {@link ByteString} data to an {@link FileOutputStream}.
   *
   * <p>Note: this implementation avoids allocating a large continuous byte[] for the entire data.
   */
  static void writeByteString(ByteString data, FileOutputStream stream) throws IOException {
    final byte[] buf = new byte[CHUNK_SIZE];
    int remainingSize = data.size();
    int offset = 0;

    while (remainingSize > 0) {
      int size = min(CHUNK_SIZE, remainingSize);
      data.substring(offset, offset + size).copyTo(buf, 0);
      stream.write(buf, 0, size);
      offset += size;
      remainingSize -= size;
    }
  }

  private FileUtil() {}
}
