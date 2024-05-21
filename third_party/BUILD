# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

java_plugin(
    name = "autovalue_plugin",
    processor_class = "com.google.auto.value.processor.AutoValueProcessor",
    deps = ["@maven//:com_google_auto_value_auto_value"],
)

java_plugin(
    name = "autooneof_plugin",
    processor_class = "com.google.auto.value.processor.AutoOneOfProcessor",
    deps = ["@maven//:com_google_auto_value_auto_value"],
)

java_library(
    name = "autovalue",
    exported_plugins = [
        ":autooneof_plugin",
        ":autovalue_plugin",
    ],
    neverlink = True,
    visibility = ["//visibility:public"],
    exports = ["@maven//:com_google_auto_value_auto_value_annotations"],
)
