// Copyright 2019 Google LLC
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

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "fcp/tracing/tracing_severity.h"
#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/idl.h"
#include "flatbuffers/reflection_generated.h"
#include "flatbuffers/util.h"

using ::reflection::BaseType;

namespace fcp {

struct TypeInfo {
  BaseType flatbuf_type;
  std::string cpp_type;
};

static std::string severity_string(const TracingSeverity tracing_severity) {
  switch (tracing_severity) {
    case TracingSeverity::kInfo:
      return "fcp::TracingSeverity::kInfo";
    case TracingSeverity::kWarning:
      return "fcp::TracingSeverity::kWarning";
    case TracingSeverity::kError:
      return "fcp::TracingSeverity::kError";
  }
}

static std::string gen_header_guard(absl::string_view output_filename) {
  std::string header_guard = absl::StrReplaceAll(
      output_filename, {{"_generated", ""}, {"/", "_"}, {".", "_"}});
  std::transform(header_guard.begin(), header_guard.end(), header_guard.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  return header_guard;
}

static std::string gen_fbs_filename(absl::string_view output_filename) {
  return absl::StrReplaceAll(output_filename, {{".h", ".fbs"}});
}

static absl::string_view gen_table_name(
    absl::string_view fully_qualified_table_name) {
  auto pos = fully_qualified_table_name.find_last_of(':');
  if (pos != std::string::npos) {
    return absl::ClippedSubstr(fully_qualified_table_name, pos + 1);
  }
  return fully_qualified_table_name;
}

static absl::string_view gen_table_namespace(
    absl::string_view fully_qualified_table_name) {
  auto pos = fully_qualified_table_name.find_last_of(':');
  if (pos != std::string::npos) {
    return absl::ClippedSubstr(fully_qualified_table_name, 0, pos + 1);
  }
  return "";
}
}  // namespace fcp

// For codegen examples, see fcp/tracing/tools/testdata.
int main(int argc, const char** argv) {
  if (argc != 4) {
    std::cerr << "Usage: tracing_traits_generator "
                 "<runtime/path/to/tracing_schema_generated.h> "
                 "<full/path/to/tracing_schema.bfbs>"
                 "<full/path/to/tracing_schema.fbs>"
              << '\n';
    return 1;
  }
  const char* generated_filename = argv[1];
  const char* bfbs_filename = argv[2];
  const char* fbs_filename = argv[3];

  // Loading binary schema file
  std::string bfbs_file;
  if (!flatbuffers::LoadFile(bfbs_filename, true, &bfbs_file)) {
    std::cerr << "Error loading FlatBuffers binary schema (bfbs) file" << '\n';
    return 2;
  }

  // Verify it, just in case:
  flatbuffers::Verifier verifier(
      reinterpret_cast<const uint8_t*>(bfbs_file.c_str()), bfbs_file.length());
  if (!reflection::VerifySchemaBuffer(verifier)) {
    std::cerr << "Error loading bfbs file" << '\n';
    return 3;
  }

  std::cout << "// Autogenerated by tracing_traits_generator, do not edit"
            << '\n';
  std::cout << '\n';

  std::string output_filename =
      absl::StrReplaceAll(generated_filename, {{"_generated", ""}});
  std::string header_guard = fcp::gen_header_guard(output_filename);
  std::cout << "#ifndef " << header_guard << '\n';
  std::cout << "#define " << header_guard << '\n';
  std::cout << '\n';

  // Workaround for inability of flatc to generate unique (path-dependent)
  // include guards. Undefining the include guard below allows
  // to include. Since all the flatc-generated schema files are wrapped
  // by the guards above, it still remains protected against multiple includes.
  std::cout << "#ifdef FLATBUFFERS_GENERATED_TRACINGSCHEMA_H_" << '\n';
  std::cout << "#undef FLATBUFFERS_GENERATED_TRACINGSCHEMA_H_" << '\n';
  std::cout << "#endif" << '\n';

  std::cout << "#include \"" << generated_filename << "\"" << '\n';
  std::cout << "#include \"absl/strings/string_view.h\"" << '\n';
  std::cout << "#include \"fcp/tracing/tracing_severity.h\""
            << '\n';
  std::cout << "#include \"fcp/tracing/tracing_traits.h\"" << '\n';
  std::cout << "#include "
               "\"flatbuffers/minireflect.h\""
            << '\n';
  std::cout << "#include "
               "\"flatbuffers/idl.h\""
            << '\n';
  std::cout << "#include "
               "\"fcp/base/platform.h\""
            << '\n';
  std::cout << '\n';

  // Reflecting over schema and enumerating tables
  auto& schema = *reflection::GetSchema(bfbs_file.c_str());
  std::cout << "namespace fcp {" << '\n';
  std::cout << '\n';

  absl::flat_hash_map<BaseType, std::string> type_map = {
      {BaseType::String, "absl::string_view"},
      {BaseType::Byte, "std::int8_t"},
      {BaseType::UByte, "std::uint8_t"},
      {BaseType::Bool, "bool"},
      {BaseType::Short, "std::int16_t"},
      {BaseType::UShort, "std::uint16_t"},
      {BaseType::Int, "std::int32_t"},
      {BaseType::UInt, "std::uint32_t"},
      {BaseType::Float, "float"},
      {BaseType::Long, "std::int64_t"},
      {BaseType::ULong, "std::uint64_t"},
      {BaseType::Double, "double"}};
  absl::flat_hash_set<std::string> tags;
  for (const reflection::Object* const o : *schema.objects()) {
    if (o->is_struct()) continue;
    std::string fully_qualified_table_name =
        absl::StrReplaceAll(o->name()->c_str(), {{".", "::"}});
    absl::string_view table_name =
        fcp::gen_table_name(fully_qualified_table_name);
    absl::string_view table_namespace =
        fcp::gen_table_namespace(fully_qualified_table_name);

    // The fields are sorted in alphabetical order, rather than the order in
    // which they should be passed to the Create method. Sort them by ID which
    // determines the order in which the generated Create method accepts them.
    // ID will be the order in which fields are declared in the table if it is
    // not explicitly specified for each field.
    std::vector<const reflection::Field*> fields_sorted;
    fields_sorted.resize(o->fields()->size());
    for (const reflection::Field* const f : *o->fields()) {
      // FlatBuffers field IDs are guaranteed to be dense:
      assert(f->id() < o->fields()->size());
      fields_sorted[f->id()] = f;
    }

    std::vector<std::pair<std::string, fcp::TypeInfo>> fields;
    for (const reflection::Field* const f : fields_sorted) {
      // Filter out deprecated fields since the Create method no longer takes
      // them as parameters.
      if (f->deprecated()) continue;
      BaseType flatbuf_type = f->type()->base_type();
      auto type_map_entry = type_map.find(flatbuf_type);
      if (type_map_entry == type_map.end()) {
        std::cerr
            << absl::StreamFormat(
                   "ERROR: %s contains unsupported type %s for field %s in "
                   "table %s",
                   fcp::gen_fbs_filename(output_filename),
                   reflection::EnumNameBaseType(flatbuf_type),
                   f->name()->c_str(), fully_qualified_table_name)
            << '\n';
        return 4;
      }
      if (f->type()->index() != -1) {
        // If the index of the type is set, it means this is a more complex
        // type, and we can learn more about the type by indexing into one of
        // the toplevel fields in the schema - either "objects" or "enums".
        // Since we do not currently support base_type of kind Union, UnionType,
        // or Object, if the index is anything other than -1, this type must be
        // an integer derived from an enum, and we can determine more
        // information by indexing into "enums". See
        // https://groups.google.com/g/flatbuffers/c/nAi8MQu3A-U.
        const reflection::Enum* enum_type =
            schema.enums()->Get(f->type()->index());
        fields.emplace_back(
            f->name()->c_str(),
            fcp::TypeInfo{
                flatbuf_type,
                // Replace '.' with '::' in the fully qualified enum name for
                // C++ compatibility.
                absl::StrReplaceAll(enum_type->name()->str(), {{".", "::"}})});
      } else {
        fields.emplace_back(
            f->name()->c_str(),
            fcp::TypeInfo{flatbuf_type, type_map_entry->second});
      }
    }

    std::cout << "template<> class TracingTraits<" << fully_qualified_table_name
              << ">: public TracingTraitsBase {" << '\n';
    std::cout << " public:" << '\n';

    fcp::TracingSeverity severity = fcp::TracingSeverity::kInfo;
    std::string tag = "";
    bool is_span = false;
    if (o->attributes() == nullptr) {
      std::cerr
          << absl::StreamFormat(
                 "ERROR: %s contains table %s without a tag. All tables must "
                 "have a tag defined.",
                 fcp::gen_fbs_filename(output_filename),
                 fully_qualified_table_name)
          << '\n';
      return 5;
    }
    for (const reflection::KeyValue* a : *o->attributes()) {
      if (a->key()->str() == "tag") {
        tag = a->value()->str();
        if (tag.size() != 4) {
          std::cerr
              << absl::StreamFormat(
                     "ERROR: %s contains table %s with tag %s of length %d. "
                     "All tables must have a tag of length 4.",
                     fcp::gen_fbs_filename(output_filename),
                     fully_qualified_table_name, tag, tag.size())
              << '\n';
          return 6;
        }
      }
      if (a->key()->str() == "warning") {
        severity = fcp::TracingSeverity::kWarning;
      }
      if (a->key()->str() == "error") {
        severity = fcp::TracingSeverity::kError;
      }
      if (a->key()->str() == "span") {
        is_span = true;
      }
    }
    if (tag.empty()) {
      std::cerr
          << absl::StreamFormat(
                 "ERROR: %s contains table %s without a tag. All tables must "
                 "have a tag defined.",
                 fcp::gen_fbs_filename(output_filename),
                 fully_qualified_table_name)
          << '\n';
      return 7;
    }

    if (!tags.insert(tag).second) {
      std::cerr
          << absl::StreamFormat(
                 "ERROR: %s contains table %s with tag %s which is already "
                 "present in the schema. All tags must be unique.",
                 fcp::gen_fbs_filename(output_filename),
                 fully_qualified_table_name, tag)
          << '\n';
      return 8;
    }

    std::cout << "  static constexpr TracingTag kTag = TracingTag(\"" << tag
              << "\");" << '\n';

    std::cout << "  static constexpr TracingSeverity kSeverity = "
              << fcp::severity_string(severity) << ";" << '\n';

    std::cout << "  static constexpr bool kIsSpan = "
              << (is_span ? "true" : "false") << ";" << '\n';

    std::cout << "  const char* Name() const override { return \""
              << fully_qualified_table_name << "\"; }" << '\n';

    std::cout << "  TracingSeverity Severity() const override {" << '\n';
    std::cout << "    return " << fcp::severity_string(severity) << ";" << '\n';
    std::cout << "  }" << '\n';
    std::cout
        << "  std::string TextFormat(const flatbuffers::DetachedBuffer& buf) "
           "const override {"
        << '\n';
    std::cout << "    return flatbuffers::FlatBufferToString(buf.data(), "
              << fully_qualified_table_name << "TypeTable());" << '\n';
    std::cout << "  }" << '\n';
    std::cout << "  std::string JsonStringFormat(const uint8_t* flatbuf_bytes) "
                 "const override {"
              << '\n';
    std::cout << "    flatbuffers::Parser parser;" << '\n';
    std::cout << "    std::string schema_file;" << '\n';
    std::cout << "    std::string fbs_file = \"" << fbs_filename << "\";"
              << '\n';
    std::cout << "    flatbuffers::LoadFile(GetDataPath(fbs_file).c_str(), "
                 "true, &schema_file);"
              << '\n';
    // Finds the directory in which the flatbuf class should look for
    // dependencies of the .fbs file
    // TODO(team) pass in tracing_schema_common to the script instead of
    // hardcoding it.
    std::cout << "    std::string schema_path_common = "
                 "GetDataPath(\"fcp/tracing/"
                 "tracing_schema_common.fbs\");"
              << '\n';
    std::cout
        << "    std::string directory_common = schema_path_common.substr(0, "
           "schema_path_common.find(\"fcp/tracing/"
           "tracing_schema_common.fbs\"));"
        << '\n';
    // Parser.parse() requires the directories passed in to have a nullptr
    std::cout << "    const char *include_directories[] = {" << '\n';
    std::cout << "                 directory_common.c_str(), nullptr};" << '\n';
    // Parse takes in the schema file and populates the FlatBufferBuilder from
    // the unique schema.
    std::cout << "    parser.Parse(schema_file.c_str(), include_directories);"
              << '\n';
    std::cout << "    std::string jsongen;" << '\n';
    // The root sets the particular table from the Flatbuffer, since flatbuffers
    // can have different tables.
    std::cout << "    parser.SetRootType(\"" << fully_qualified_table_name
              << "\");" << '\n';
    std::cout << "    GenText(parser, flatbuf_bytes, &jsongen);" << '\n';
    std::cout << "    return jsongen;" << '\n';
    std::cout << "  }" << '\n';
    std::cout << "  static flatbuffers::Offset<" << fully_qualified_table_name;
    std::cout << "> Create(";
    for (const auto& [name, type] : fields) {
      std::cout << type.cpp_type << " " << name << ", ";
    }
    std::cout << "flatbuffers::FlatBufferBuilder* fbb) {" << '\n';

    // Strings require special handling because the Create method takes an
    // Offset<String>. Copy each provided string view into an Offset<String>.
    for (const auto& [name, type] : fields) {
      if (type.flatbuf_type == BaseType::String) {
        std::cout << "    auto " << name << "__ = fbb->CreateString(" << name
                  << ".data(), " << name << ".size()"
                  << ");" << '\n';
      }
    }

    std::cout << "    return " << table_namespace << "Create" << table_name
              << "(";
    std::cout << "*fbb";
    for (const auto& [name, type] : fields) {
      const char* suffix = (type.flatbuf_type == BaseType::String) ? "__" : "";
      std::cout << ", " << name << suffix;
    }
    std::cout << ");" << '\n';
    std::cout << "  }" << '\n';

    // MakeTuple helper, which allows to generate std::tuple from a table.
    std::string tuple_type = "std::tuple<";
    std::string make_tuple_args;
    for (const auto& [name, type] : fields) {
      if (!make_tuple_args.empty()) {
        tuple_type += ", ";
        make_tuple_args += ", ";
      }
      make_tuple_args += "table->" + name + "()";
      if (type.flatbuf_type == BaseType::String) {
        tuple_type += "std::string";
        make_tuple_args += "->str()";
      } else {
        tuple_type += std::string(type.cpp_type);
      }
    }
    tuple_type += ">";

    std::cout << "  using TupleType = " << tuple_type << ";" << '\n';
    std::cout << "  static TupleType MakeTuple(const "
              << fully_qualified_table_name << "* table) {" << '\n';
    std::cout << "    return std::make_tuple(" << make_tuple_args << ");"
              << '\n';
    std::cout << "  }" << '\n';

    std::cout << "};" << '\n';
    std::cout << "static internal::TracingTraitsRegistrar<"
              << fully_qualified_table_name << "> registrar_" << table_name
              << ";" << '\n';
  }
  std::cout << "} // namespace fcp" << '\n';
  std::cout << '\n';
  std::cout << "#endif  // " << header_guard << '\n';
  return 0;
}
