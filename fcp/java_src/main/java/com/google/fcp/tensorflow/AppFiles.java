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

import java.io.File;
import java.io.IOException;

/** A class which gives access to application files. */
public final class AppFiles {

  private final File virtualCacheDir;

  public AppFiles(String dirRoot) {
    virtualCacheDir = new File(dirRoot, "cache");
    virtualCacheDir.mkdirs();
  }

  public File createTempFile(String prefix, String suffix) throws IOException {
    return File.createTempFile(prefix, suffix, virtualCacheDir);
  }
}
