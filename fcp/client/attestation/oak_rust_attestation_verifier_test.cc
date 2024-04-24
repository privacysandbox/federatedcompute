#include "fcp/client/attestation/oak_rust_attestation_verifier.h"

#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/escaping.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "fcp/base/compression.h"
#include "fcp/base/digest.h"
#include "fcp/client/rust/oak_attestation_verification_ffi.h"
#include "fcp/confidentialcompute/cose.h"
#include "fcp/confidentialcompute/crypto.h"
#include "fcp/protos/confidentialcompute/access_policy.pb.h"
#include "fcp/protos/confidentialcompute/verification_record.pb.h"
#include "fcp/protos/federatedcompute/confidential_aggregations.pb.h"
#include "fcp/testing/testing.h"
#include "proto/attestation/endorsement.pb.h"
#include "proto/attestation/evidence.pb.h"
#include "proto/attestation/reference_value.pb.h"
#include "proto/attestation/verification.pb.h"
#include "proto/digest.pb.h"

namespace fcp::client::attestation {
namespace {
using ::fcp::confidential_compute::OkpCwt;
using ::google::internal::federatedcompute::v1::ConfidentialEncryptionConfig;
using ::oak::attestation::v1::ReferenceValues;
using ::testing::_;
using ::testing::Each;
using ::testing::FieldsAre;
using ::testing::Gt;
using ::testing::HasSubstr;
using ::testing::SizeIs;

// Tests the case where default values are given to the verifier. Verification
// should not succeed in this case, since the Rust Oak Attestation Verification
// library will complain that required values are missing.
//
// This validates that we at least correctly call into the Rust Oak Attestation
// Verification library and get a result back, but it doesn't actually test
// the actual verification logic.
TEST(OakRustAttestationTest, DefaultValuesDoNotVerifySuccessfully) {
  // Generate a new public key, which we'll pass to the client in the
  // ConfidentialEncryptionConfig. We'll use the decryptor from which the public
  // key was generated to validate the encrypted payload at the end of the test.
  fcp::confidential_compute::MessageDecryptor decryptor;
  auto encoded_public_key =
      decryptor
          .GetPublicKey(
              [](absl::string_view payload) { return "fakesignature"; }, 0)
          .value();
  absl::StatusOr<OkpCwt> parsed_public_key = OkpCwt::Decode(encoded_public_key);
  ASSERT_OK(parsed_public_key);
  ASSERT_TRUE(parsed_public_key->public_key.has_value());

  // Note: we don't specify any attestation evidence nor attestation
  // endorsements in the encryption config, since we can't generate valid
  // attestations in a test anyway.
  ConfidentialEncryptionConfig encryption_config;
  encryption_config.set_public_key(encoded_public_key);
  // Populate an empty Evidence proto.
  encryption_config.mutable_attestation_evidence();

  // Use an empty ReferenceValues input.
  ReferenceValues reference_values;
  OakRustAttestationVerifier verifier(reference_values, {},
                                      LogPrettyPrintedVerificationRecord);

  // The verification should fail, since neither reference values nor the
  // evidence are valid.
  auto result = verifier.Verify(absl::Cord(""), encryption_config);
  EXPECT_THAT(result.status(), IsCode(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(result.status().message(),
              HasSubstr("Attestation verification failed"));
}

ConfidentialEncryptionConfig GetKnownValidEncryptionConfig() {
  return PARSE_TEXT_PROTO(R"pb(
    public_key: "\204C\241\001&\240XH\243\004\032f\r\335A\006\032f\004\242\301:\000\001\000\000X4\245\001\001\002DH)4\264\003:\000\001\000\000 \004!X \3562\356Py\304\002n?@\360\353Jc\245mE$q\324O\257\033\3778f\333s\254\252\355yX@\245\330\311%\345\377\364w\203|Z\033\t\263\226\232f\350\031m\230Gm\241M\237\023\025\220-\036\370\tW\315 .x&\211\311\216\276\316\350\002\350\302R\024B\243\312\306O\341\214\363\232d\342\177u7"
    attestation_evidence {
      root_layer {
        platform: AMD_SEV_SNP
        remote_attestation_report: "\002\000\000\000\000\000\000\000\000\000\003\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\001\000\000\000\003\000\000\000\000\000\024\321\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000<a\022}?\003\237\216\247\372\354)\366c\327\323\263\272kV\207N\177cRSe1U\314\202\374\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\264{\300\362g?r\225\343\202\013\021Q\247E\363(\013\211\270\213\2719\375\353\367\034p\253%\000)\356\246\337\264\247\245d?\206\333\3033\301\302I)\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\203\016\362\225\253\237\250\275*}\022>\337B\206\006u\366\354\232\355\376\035\016\213\306\342\351\206\244C/\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377\377\003\000\000\000\000\000\024\321\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\343\014\".\276b\337\n\nq\203\205Z\352\003\330\315\344\210P8\214\002\274\306\347&^\223\343(\211\330\236\326\360\nny/\340gy\270\260\354@NF\205\300ADkP{\237\370U\325>U\0071\003\000\000\000\000\000\024\321\0207\001\000\0207\001\000\003\000\000\000\000\000\024\321\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\3563\202=\rW]\021\236\335\342\226\2330\377V\361\242\'\233\000\0226.u[.+\303\372\370rU\237\033\271\332\036n\350v\212_lu\367|\267\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000F\250\304E \235 \201\250d\314L1SI\367\333t\335\311\030\232!\224\237\035v\275\2119\205\231\023\270\335q\020\222\244H\304\242,:\346?*\276\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000"
        eca_public_key: "\247\001\002\002T\242E3\332\242=\225%\360\255\310\313yy0s\323\376T\n\003&\004\201\002 \001!X j\347x\317\010jw\212\251\221d\023g\366\033\235\366\377NAp\201:\004\332\000\374u%\204\224\277\"X \333\265\213K@\334\344\360\342\212J\002<\025l\223\242\220w\2449\226\303\025\343:S\321;\0359]"
      }
      layers {
        eca_certificate: "\204C\241\001&\241\004RAsymmetricECDSA256Y\001\363\245\001x(a24533daa23d9525f0adc8cb79793073d3fe540a\002x(bdad38f82eb2eda797d46af92597b6eb30fde2d7:\000GDWXf\247\001\002\002T\275\2558\370.\262\355\247\227\324j\371%\227\266\3530\375\342\327\003&\004\201\002 \001!X %\264C\210\212\347\0339\025\215\332\206@\340\331\303\370\3015\001\356\271\335l\013\267\266n\245\013\242m\"X R\000\255\345r\203\231\345A&iB\242\3716i\200\270\024\231\203\357j\277\307\250\215xk\303\t\010:\000GDXB \000:\000GDZ\247:\000GD`\241:\000GDkX \273\024\236X\036\330X\324&\232\317\204L\251\316\260\001b\362\342\252. a\007$b\240^\014\207C:\000GDa\241:\000GDkX +\230Xm\231\005\246\005\302\225\327|a\350\317\322\002z\345\270\240N\357\251\001\2046\366\255\021B\227:\000GDlmconsole=ttyS0:\000GDb\241:\000GDkX L\320 \202\r\246c\006?A\205\312\024\247\350\003\315|\234\241H<d\3506\333\204\006\004\266\372\301:\000GDc\241:\000GDkX \000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000:\000GDd\241:\000GDkX (4\213A.\201\255Z\225k\366Z n\342\006\315\315q*\220j\2154\263+zl\221\274\026\031:\000GDe\241:\000GDkX \333\254\312\347\277\277\000n+\206#\250/\032_\315\242\352\003\222#<&\261\203V\263\274\372\3021\353X@\037\n\'\006\313\317\025\014\202-\202\314\rs%\353dc\240z\rT\364\001+\271\001%6\2555P\374T\220\027\017M\224\343h\236J\321ag_\354\002V&\206\240\365\210_\202\017n \266\243\3024"
      }
      application_keys {
        encryption_public_key_certificate: "\204C\241\001&\241\004RAsymmetricECDSA256X\351\245\001x(bdad38f82eb2eda797d46af92597b6eb30fde2d7\002x(6fd19b1a5aa9086d68d5db459531d4a0e538b81e:\000GDWXD\246\001\001\002To\321\233\032Z\251\010mh\325\333E\2251\324\240\3458\270\036\0038\036\004\201\005 \004!X \304\375\276g+$\373e\262\263\324\361\246\375 0d\300\030R\236,{\206\207\372T\312\332\373\276y:\000GDXB \000:\000GD[\242:\000GDf\241:\000GDkX \026>\241\310\354#us\250\032!\257\271\254XXgu\345tW\220.u\365B|\277\025]*\336:\000GDi\241:\000GDk@X@\037\316\020\317-\352\352\031\352!.eEa\343\3204n\002\370\034v\371\235|/0t\035\250\036\316\246)\257\260\000\330\232J\333Cqc\214\233\2033\031)IB\370\313\3331@\355\035lTw\241\316"
        signing_public_key_certificate: "\204C\241\001&\241\004RAsymmetricECDSA256Y\001\013\245\001x(bdad38f82eb2eda797d46af92597b6eb30fde2d7\002x(2a9a332197a4909c2ffdd4b99a341089ecd3a7a1:\000GDWXf\247\001\002\002T*\2323!\227\244\220\234/\375\324\271\2324\020\211\354\323\247\241\003&\004\201\002 \001!X %\376\3517\361\203\264\013q%\302\236Q\006\022\213\264\302\374\201\266@\332\347\336{\327\200\2376\347\233\"X \371.\261X\252yHk=\364x\030V\262\335t6:> \257\332\247u\341\374\261\220\027\374\353\355:\000GDXB \000:\000GD[\242:\000GDf\241:\000GDkX \026>\241\310\354#us\250\032!\257\271\254XXgu\345tW\220.u\365B|\277\025]*\336:\000GDi\241:\000GDk@X@-\326-\250r\002\324rE\004\300\367\222\325\230W\025Q\340\247\233\271\034~0P\003\203\2375\301\233\353\372z\227\035n\225\007l\021i\202\027l\030\242\332\202LB@-\3348\230\250\257\341e\t\007\'"
      }
    }
    attestation_endorsements {
      oak_restricted_kernel {
        root_layer {
          tee_certificate: "0\202\005M0\202\002\374\240\003\002\001\002\002\001\0000F\006\t*\206H\206\367\r\001\001\n09\240\0170\r\006\t`\206H\001e\003\004\002\002\005\000\241\0340\032\006\t*\206H\206\367\r\001\001\0100\r\006\t`\206H\001e\003\004\002\002\005\000\242\003\002\0010\243\003\002\001\0010{1\0240\022\006\003U\004\013\014\013Engineering1\0130\t\006\003U\004\006\023\002US1\0240\022\006\003U\004\007\014\013Santa Clara1\0130\t\006\003U\004\010\014\002CA1\0370\035\006\003U\004\n\014\026Advanced Micro Devices1\0220\020\006\003U\004\003\014\tSEV-Milan0\036\027\r240303194456Z\027\r310303194456Z0z1\0240\022\006\003U\004\013\014\013Engineering1\0130\t\006\003U\004\006\023\002US1\0240\022\006\003U\004\007\014\013Santa Clara1\0130\t\006\003U\004\010\014\002CA1\0370\035\006\003U\004\n\014\026Advanced Micro Devices1\0210\017\006\003U\004\003\014\010SEV-VCEK0v0\020\006\007*\206H\316=\002\001\006\005+\201\004\000\"\003b\000\004D\2362\254\336[\222\316(\236a\006\337\324z\2508v3Xu\351\366\316q\320\373\350tTc\363\353p\323\033\n\205>\221\336\277n\232eT\221-\241\002\335c6\270\030\252\354\250\247\324\270\366F\013[\306\363=k\371\031\306\266\240\243\307D\317\034c^8\'\027\343\233\263oO\326Z\220\363y?\210\243\202\001\0270\202\001\0230\020\006\t+\006\001\004\001\234x\001\001\004\003\002\001\0000\027\006\t+\006\001\004\001\234x\001\002\004\n\026\010Milan-B00\021\006\n+\006\001\004\001\234x\001\003\001\004\003\002\001\0030\021\006\n+\006\001\004\001\234x\001\003\002\004\003\002\001\0000\021\006\n+\006\001\004\001\234x\001\003\004\004\003\002\001\0000\021\006\n+\006\001\004\001\234x\001\003\005\004\003\002\001\0000\021\006\n+\006\001\004\001\234x\001\003\006\004\003\002\001\0000\021\006\n+\006\001\004\001\234x\001\003\007\004\003\002\001\0000\021\006\n+\006\001\004\001\234x\001\003\003\004\003\002\001\0240\022\006\n+\006\001\004\001\234x\001\003\010\004\004\002\002\000\3210M\006\t+\006\001\004\001\234x\001\004\004@\343\014\".\276b\337\n\nq\203\205Z\352\003\330\315\344\210P8\214\002\274\306\347&^\223\343(\211\330\236\326\360\nny/\340gy\270\260\354@NF\205\300ADkP{\237\370U\325>U\00710F\006\t*\206H\206\367\r\001\001\n09\240\0170\r\006\t`\206H\001e\003\004\002\002\005\000\241\0340\032\006\t*\206H\206\367\r\001\001\0100\r\006\t`\206H\001e\003\004\002\002\005\000\242\003\002\0010\243\003\002\001\001\003\202\002\001\000[\302\311J\331!6f~\037\332\270C\005\nP\013\001\215\211\333\373\002\2452XeE\372\255\256T\216\325\336,rt_\010\222\314\2313\233t\200c\016?~\001x\305\321\007p3\366\021[\372\037\370\356Bf\3727\r7b3md\364\301Y\2079y\"\rF\345\327\326\306Q\337\254Y\255:\201It\246{\311\235.N<\271\230i\321U\353\230\371;\203\014\224\342\313\251\372\031\335,Y\242n\267\204{{@)\010z(\034\254\214\351h\254\241\225\010;Y\235\204\2506\363w\023\216\036\3662\213\306\203\347m\010J\275\371\237\2518\332\030\323D\226\224\214\361N\373\014b6\347\322\300\255\353\266Wa|\024\223\243I!\202\034\367\025\337\365\232\345\212\002\2573Bb\0211\316RLV\253\000\0063\372\347\355\344\355\225r\n\233\005\335\327\2374R!\367(2\277\251\'\364\332:\2427\217\311\310Q*\177\255\252\034L\274\0206\020\375 \024\020km1g\326\rb\245j\3232\376\r\365\276\230\032\213\277r\346\340>\365Vy\364\033=\354\362\336\265\316\306}\215\204\00047X\rz\240\373\311T\001Br\245y\230\263\373\301=\236\254c;\257\324?\323e\330\311\024R\243\301I\226\033v\035\310\376\277\216O\341\344\0077\33233~\364=\244\251\250K\000x\317\235\331,\264\244 \234\032\263&\205\305E\250+\0101[\363\201iD\235\033\254y%\343;\221\352\220y\355-N\r\317@\233\364\305A.\252\221\347y\364\004\241\'\263i\216_\213^\345\247\374Bq\206\316HD\242Q\033ch\r<\207\3044\376\224\024*1\234\200\333\301\031\310\224\231\005\005\021\024\"T\307\244AZ\303\211w#<\373\243\t\2366\023n\344\311=u\023\330\222\267\376\302\336\204\270kRB\'%9\317\276\321@\321 \000d\242\335\331\277<\256\014\025"
        }
      }
    }
  )pb");
}

ReferenceValues GetKnownValidReferenceValues() {
  return PARSE_TEXT_PROTO(R"pb(
    oak_restricted_kernel {
      root_layer {
        amd_sev {
          min_tcb_version {}
          stage0 {
            digests {
              digests {
                sha2_384: "\264{\300\362g?r\225\343\202\013\021Q\247E\363(\013\211\270\213\2719\375\353\367\034p\253%\000)\356\246\337\264\247\245d?\206\333\3033\301\302I)"
              }
            }
          }
        }
      }
      kernel_layer {
        kernel {
          digests {
            image {
              digests {
                sha2_256: "\273\024\236X\036\330X\324&\232\317\204L\251\316\260\001b\362\342\252. a\007$b\240^\014\207C"
              }
            }
            setup_data {
              digests {
                sha2_256: "L\320 \202\r\246c\006?A\205\312\024\247\350\003\315|\234\241H<d\3506\333\204\006\004\266\372\301"
              }
            }
          }
        }
        kernel_cmd_line_text { string_literals { value: "console=ttyS0" } }
        init_ram_fs {
          digests {
            digests {
              sha2_256: "\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000"
            }
          }
        }
        memory_map {
          digests {
            digests {
              sha2_256: "(4\213A.\201\255Z\225k\366Z n\342\006\315\315q*\220j\2154\263+zl\221\274\026\031"
            }
          }
        }
        acpi {
          digests {
            digests {
              sha2_256: "\333\254\312\347\277\277\000n+\206#\250/\032_\315\242\352\003\222#<&\261\203V\263\274\372\3021\353"
            }
          }
        }
      }
      application_layer {
        binary {
          digests {
            digests {
              sha2_256: "\x16\x3e\xa1\xc8\xec\x23\x75\x73\xa8\x1a\x21\xaf\xb9\xac\x58\x58\x67\x75\xe5\x74\x57\x90\x2e\x75\xf5\x42\x7c\xbf\x15\x5d\x2a\xde"
            }
          }
        }
        # The binary doesn't use any configuration, so nothing to check.
        configuration { skip {} }
      }
    }
  )pb");
}

ReferenceValues GetSkipAllReferenceValues() {
  return PARSE_TEXT_PROTO(R"pb(
    oak_restricted_kernel {
      root_layer {
        amd_sev {
          min_tcb_version {}
          stage0 { skip {} }
        }
      }
      kernel_layer {
        kernel { skip {} }
        kernel_cmd_line_text { skip {} }
        init_ram_fs { skip {} }
        memory_map { skip {} }
        acpi { skip {} }
      }
      application_layer {
        binary { skip {} }
        configuration { skip {} }
      }
    }
  )pb");
}

TEST(OakRustAttestationTest,
     KnownValidEncryptionConfigAndValidPolicyInAllowlist) {
  ConfidentialEncryptionConfig encryption_config =
      GetKnownValidEncryptionConfig();
  ReferenceValues reference_values = GetKnownValidReferenceValues();

  // Create a valid access policy proto with some non-default content.
  confidentialcompute::DataAccessPolicy access_policy = PARSE_TEXT_PROTO(R"pb(
    transforms {
      src: 0
      application { tag: "foo" }
    }
  )pb");
  auto access_policy_bytes = access_policy.SerializeAsString();

  // This verifier will only accept attestation evidence matching the reference
  // values defined above, and will only accept the given access policy.
  OakRustAttestationVerifier verifier(
      reference_values,
      {absl::BytesToHexString(ComputeSHA256(access_policy_bytes))},
      LogPrettyPrintedVerificationRecord);

  // Ensure that the verification succeeds.
  auto result =
      verifier.Verify(absl::Cord(access_policy_bytes), encryption_config);
  ASSERT_OK(result);
}

TEST(OakRustAttestationTest, KnownValidEncryptionConfigAndMismatchingPolicy) {
  ConfidentialEncryptionConfig encryption_config =
      GetKnownValidEncryptionConfig();
  ReferenceValues reference_values = GetKnownValidReferenceValues();

  // Create a valid access policy proto with some non-default content.
  confidentialcompute::DataAccessPolicy disallowed_access_policy =
      PARSE_TEXT_PROTO(R"pb(
        transforms {
          src: 0
          application { tag: "bar" }
        }
      )pb");
  auto disallowed_access_policy_bytes =
      disallowed_access_policy.SerializeAsString();

  // This verifier will not accept any inputs, since the policy allowlist
  // doesn't match the actual policy.
  OakRustAttestationVerifier verifier(reference_values,
                                      {"mismatching policy hash"},
                                      LogPrettyPrintedVerificationRecord);

  // Ensure that the verification *does not* succeed.
  auto result = verifier.Verify(absl::Cord(disallowed_access_policy_bytes),
                                encryption_config);
  EXPECT_THAT(result.status(), IsCode(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(result.status().message(),
              HasSubstr("Data access policy not in allowlist"));
}

TEST(OakRustAttestationTest,
     KnownValidEncryptionConfigAndAndValidPolicyWithEmptyPolicyAllowlist) {
  ConfidentialEncryptionConfig encryption_config =
      GetKnownValidEncryptionConfig();
  ReferenceValues reference_values = GetKnownValidReferenceValues();

  // Create a valid access policy proto with some non-default content.
  confidentialcompute::DataAccessPolicy access_policy = PARSE_TEXT_PROTO(R"pb(
    transforms {
      src: 0
      application { tag: "foo" }
    }
  )pb");
  auto access_policy_bytes = access_policy.SerializeAsString();

  // This verifier will not accept any inputs, since the policy allowlist is
  // empty.
  OakRustAttestationVerifier verifier(reference_values, {},
                                      LogPrettyPrintedVerificationRecord);

  // Ensure that the verification *does not* succeed.
  auto result =
      verifier.Verify(absl::Cord(access_policy_bytes), encryption_config);
  EXPECT_THAT(result.status(), IsCode(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(result.status().message(),
              HasSubstr("Data access policy not in allowlist"));
}

TEST(OakRustAttestationTest,
     KnownValidEncryptionConfigAndAndEmptyPolicyWithEmptyPolicyAllowlist) {
  ConfidentialEncryptionConfig encryption_config =
      GetKnownValidEncryptionConfig();
  ReferenceValues reference_values = GetKnownValidReferenceValues();

  // This verifier will not accept any inputs, since the policy allowlist is
  // empty.
  OakRustAttestationVerifier verifier(reference_values, {},
                                      LogPrettyPrintedVerificationRecord);

  // Ensure that the verification *does not* succeed, since an empty access
  // policy string still has to match an allowlist entry.
  auto result = verifier.Verify(absl::Cord(""), encryption_config);
  EXPECT_THAT(result.status(), IsCode(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(result.status().message(),
              HasSubstr("Data access policy not in allowlist"));
}

TEST(OakRustAttestationTest,
     KnownEncryptionConfigAndMismatchingReferencevalues) {
  ConfidentialEncryptionConfig encryption_config =
      GetKnownValidEncryptionConfig();
  ReferenceValues reference_values = GetKnownValidReferenceValues();
  // Mess with the application layer digest value to ensure it won't match the
  // values in the ConfidentialEncryptionConfig.
  (*reference_values.mutable_oak_restricted_kernel()
        ->mutable_application_layer()
        ->mutable_binary()
        ->mutable_digests()
        ->mutable_digests(0)
        ->mutable_sha2_256())[0] += 1;

  // Create a valid access policy proto with some non-default content.
  confidentialcompute::DataAccessPolicy access_policy = PARSE_TEXT_PROTO(R"pb(
    transforms {
      src: 0
      application { tag: "bar" }
    }
  )pb");
  auto access_policy_bytes = access_policy.SerializeAsString();

  // This verifier will not accept the encryption config provided, due to the
  // mismatching digest.
  OakRustAttestationVerifier verifier(
      reference_values,
      {absl::BytesToHexString(ComputeSHA256(access_policy_bytes))},
      LogPrettyPrintedVerificationRecord);

  // Ensure that the verification *does not* succeed.
  auto result =
      verifier.Verify(absl::Cord(access_policy_bytes), encryption_config);
  EXPECT_THAT(result.status(), IsCode(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(result.status().message(),
              HasSubstr("Attestation verification failed"));
}

TEST(OakRustAttestationTest, KnownEncryptionConfigAndEmptyReferencevalues) {
  ConfidentialEncryptionConfig encryption_config =
      GetKnownValidEncryptionConfig();

  // Create a valid access policy proto with some non-default content.
  confidentialcompute::DataAccessPolicy access_policy = PARSE_TEXT_PROTO(R"pb(
    transforms {
      src: 0
      application { tag: "bar" }
    }
  )pb");
  auto access_policy_bytes = access_policy.SerializeAsString();

  // This verifier will not accept the encryption config provided, due to the
  // reference values being invalid (an empty, uninitialized proto).
  OakRustAttestationVerifier verifier(
      ReferenceValues(),
      {absl::BytesToHexString(ComputeSHA256(access_policy_bytes))},
      LogPrettyPrintedVerificationRecord);

  // Ensure that the verification *does not* succeed.
  auto result =
      verifier.Verify(absl::Cord(access_policy_bytes), encryption_config);
  EXPECT_THAT(result.status(), IsCode(absl::StatusCode::kFailedPrecondition));
  EXPECT_THAT(result.status().message(),
              HasSubstr("Attestation verification failed"));
}

// Tests whether the AttestationVerificationRecord emitted by the
// OakRustAttestationVerifier contains sufficient information to allow someone
// to re-do the verification (e.g. on their computer, by calling into the Oak
// Attestation Verification library themselves).
TEST(OakRustAttestationTest,
     AttestationVerificationRecordContainsEnoughInfoToReplayVerification) {
  // First, perform a normal verification pass using known-good values.
  ConfidentialEncryptionConfig encryption_config =
      GetKnownValidEncryptionConfig();
  ReferenceValues reference_values = GetKnownValidReferenceValues();

  // Create a valid access policy proto with some non-default content.
  confidentialcompute::DataAccessPolicy access_policy = PARSE_TEXT_PROTO(R"pb(
    transforms {
      src: 0
      application { tag: "foo" }
    }
  )pb");
  auto access_policy_bytes = access_policy.SerializeAsString();

  confidentialcompute::AttestationVerificationRecord verification_record;
  // This verifier will only accept attestation evidence matching the reference
  // values defined above, and will only accept the given access policy.
  OakRustAttestationVerifier verifier(
      reference_values,
      {absl::BytesToHexString(ComputeSHA256(access_policy_bytes))},
      [&verification_record](
          confidentialcompute::AttestationVerificationRecord record) {
        verification_record = record;
      });

  // Ensure that the verification succeeds.
  auto result =
      verifier.Verify(absl::Cord(access_policy_bytes), encryption_config);
  ASSERT_OK(result);

  // Ensure that the verification record logger was called and provided the
  // relevant information.
  EXPECT_THAT(verification_record.attestation_evidence(),
              EqualsProto(encryption_config.attestation_evidence()));
  EXPECT_THAT(verification_record.attestation_endorsements(),
              EqualsProto(encryption_config.attestation_endorsements()));
  EXPECT_THAT(verification_record.data_access_policy(),
              EqualsProto(access_policy));

  // Now, let's act like we're re-verifying the information in the
  // AttestationVerificationRecord in an offline fashion, by calling directly
  // into the Rust-based Oak Attestation Verification library.

  // First, we'll pass the attestation evidence to the verification library
  // using a ReferenceValues proto that skips all actual checks. This allows us
  // to access the information embedded within the attestation evidence more
  // easily.
  absl::StatusOr<oak::attestation::v1::AttestationResults>
      raw_attestation_results = fcp::client::rust::
          oak_attestation_verification_ffi::VerifyAttestation(
              absl::Now(), verification_record.attestation_evidence(),
              verification_record.attestation_endorsements(),
              GetSkipAllReferenceValues());
  ASSERT_OK(raw_attestation_results);
  ASSERT_EQ(raw_attestation_results->status(),
            oak::attestation::v1::AttestationResults::STATUS_SUCCESS)
      << raw_attestation_results->reason();

  // Then, let's create a ReferenceValues proto that requires the attestation
  // evidence to be rooted in the AMD SEV-SNP hardware root of trust, and which
  // requires each layer of the attestation evidence to match the exact binary
  // digests that were earlier reported in the `AttestationResults`.
  ReferenceValues reference_values_from_extracted_evidence;
  // Populate root layer values.
  reference_values_from_extracted_evidence.mutable_oak_restricted_kernel()
      ->mutable_root_layer()
      ->mutable_amd_sev()
      ->mutable_min_tcb_version();
  *reference_values_from_extracted_evidence.mutable_oak_restricted_kernel()
       ->mutable_root_layer()
       ->mutable_amd_sev()
       ->mutable_stage0()
       ->mutable_digests()
       ->add_digests()
       ->mutable_sha2_384() = raw_attestation_results->extracted_evidence()
                                  .oak_restricted_kernel()
                                  .root_layer()
                                  .sev_snp()
                                  .initial_measurement();
  // Populate kernel layer values.
  *reference_values_from_extracted_evidence.mutable_oak_restricted_kernel()
       ->mutable_kernel_layer()
       ->mutable_kernel()
       ->mutable_digests()
       ->mutable_image()
       ->add_digests() = raw_attestation_results->extracted_evidence()
                             .oak_restricted_kernel()
                             .kernel_layer()
                             .kernel_image();
  *reference_values_from_extracted_evidence.mutable_oak_restricted_kernel()
       ->mutable_kernel_layer()
       ->mutable_kernel()
       ->mutable_digests()
       ->mutable_setup_data()
       ->add_digests() = raw_attestation_results->extracted_evidence()
                             .oak_restricted_kernel()
                             .kernel_layer()
                             .kernel_setup_data();
  *reference_values_from_extracted_evidence.mutable_oak_restricted_kernel()
       ->mutable_kernel_layer()
       ->mutable_kernel_cmd_line_text()
       ->mutable_string_literals()
       ->add_value() = raw_attestation_results->extracted_evidence()
                           .oak_restricted_kernel()
                           .kernel_layer()
                           .kernel_raw_cmd_line();
  *reference_values_from_extracted_evidence.mutable_oak_restricted_kernel()
       ->mutable_kernel_layer()
       ->mutable_init_ram_fs()
       ->mutable_digests()
       ->add_digests() = raw_attestation_results->extracted_evidence()
                             .oak_restricted_kernel()
                             .kernel_layer()
                             .init_ram_fs();
  *reference_values_from_extracted_evidence.mutable_oak_restricted_kernel()
       ->mutable_kernel_layer()
       ->mutable_memory_map()
       ->mutable_digests()
       ->add_digests() = raw_attestation_results->extracted_evidence()
                             .oak_restricted_kernel()
                             .kernel_layer()
                             .memory_map();
  *reference_values_from_extracted_evidence.mutable_oak_restricted_kernel()
       ->mutable_kernel_layer()
       ->mutable_acpi()
       ->mutable_digests()
       ->add_digests() = raw_attestation_results->extracted_evidence()
                             .oak_restricted_kernel()
                             .kernel_layer()
                             .acpi();
  // Populate application layer values.
  *reference_values_from_extracted_evidence.mutable_oak_restricted_kernel()
       ->mutable_application_layer()
       ->mutable_binary()
       ->mutable_digests()
       ->add_digests() = raw_attestation_results->extracted_evidence()
                             .oak_restricted_kernel()
                             .application_layer()
                             .binary();
  // Add a digest for the application layer config, if the extracted evidence
  // indicates there was an application layer config, otherwise skip the
  // application layer config check since the application doesn't have a config.
  if (raw_attestation_results->extracted_evidence()
          .oak_restricted_kernel()
          .application_layer()
          .config()
          .ByteSizeLong() > 0) {
    *reference_values_from_extracted_evidence.mutable_oak_restricted_kernel()
         ->mutable_application_layer()
         ->mutable_configuration()
         ->mutable_digests()
         ->add_digests() = raw_attestation_results->extracted_evidence()
                               .oak_restricted_kernel()
                               .application_layer()
                               .config();
  } else {
    reference_values_from_extracted_evidence.mutable_oak_restricted_kernel()
        ->mutable_application_layer()
        ->mutable_configuration()
        ->mutable_skip();
  }

  // Lastly, let's verify that verifying the attestation evidence reported in
  // the AttestationVerificationRecord using the now fully-specified
  // ReferenceValues still results in a successful verification.
  //
  // This shows that the data in the AttestationVerificationRecord was indeed
  // valid (as long as we can assume that the Oak Attestation Verification
  // library is implemented correctly).
  raw_attestation_results =
      fcp::client::rust::oak_attestation_verification_ffi::VerifyAttestation(
          absl::Now(), verification_record.attestation_evidence(),
          verification_record.attestation_endorsements(),
          reference_values_from_extracted_evidence);
  ASSERT_OK(raw_attestation_results);
  EXPECT_EQ(raw_attestation_results->status(),
            oak::attestation::v1::AttestationResults::STATUS_SUCCESS)
      << raw_attestation_results->reason();

  // The attestation verification passed, as expected. We can also show that the
  // ReferenceValues proto we constructed from the extracted attestation
  // evidence is effectively the same as the "known good ReferenceValues" we use
  // in the other tests.
  EXPECT_THAT(reference_values_from_extracted_evidence,
              EqualsProto(GetKnownValidReferenceValues()));

  // At this point someone performing an offline re-verification could start
  // looking at the specific binaries that the attestation evidence attested to,
  // by looking them up using the binary digests in the ReferenceValues we
  // constructed above.
}

// Verifies that the LogSerializedVerificationRecord correctly chunks up and
// encodes the serialized record data.
TEST(LogSerializedVerificationRecordTest,
     DecodingChunkedMessagesResultsInOriginalRecord) {
  // Create a verification record with a good amount of data in it.
  confidentialcompute::AttestationVerificationRecord record;
  auto encryption_config = GetKnownValidEncryptionConfig();
  *record.mutable_attestation_evidence() =
      encryption_config.attestation_evidence();
  *record.mutable_attestation_endorsements() =
      encryption_config.attestation_endorsements();
  *record.mutable_data_access_policy()
       ->add_transforms()
       ->mutable_application()
       ->mutable_tag() = "some tag";

  // Call the LogSerializedVerificationRecord function (or rather, the internal
  // variant which allows us to inspect each of the chunks it would log), which
  // is expected to emit a number of chunks.
  std::vector<std::pair<std::string, bool>> encoded_record_data;
  internal::LogSerializedVerificationRecordWith(
      record, [&encoded_record_data](absl::string_view message_chunk,
                                     bool enclose_with_brackets) {
        encoded_record_data.push_back(
            std::make_pair(std::string(message_chunk), enclose_with_brackets));
      });

  // We expect to see at least 15 log message chunks.
  EXPECT_THAT(encoded_record_data, SizeIs(Gt(15)));
  // The first message chunk is expected to contain the following unenclosed
  // string.
  EXPECT_THAT(
      encoded_record_data.front(),
      FieldsAre(
          "This device is contributing data via the confidential aggregation "
          "protocol. The attestation verification record follows.",
          false));
  // The last chunk is expected to be an empty enclosed string, unambiguously
  // indicating the end of verification record stream.
  EXPECT_THAT(encoded_record_data.back(), FieldsAre("", true));
  encoded_record_data.erase(encoded_record_data.begin());
  encoded_record_data.erase(encoded_record_data.end() - 1);

  // The chunks in between are expected to contain enclosed data. Let's verify
  // that, and extract the inner data.
  std::string base64_record_data;
  EXPECT_THAT(encoded_record_data, Each(FieldsAre(_, true)));
  for (const auto& message_chunk : encoded_record_data) {
    base64_record_data += message_chunk.first;
  }

  // Now let's base64-decode that data, and verify that it can be parsed and
  // results in the record we started with at the top of the test.
  std::string decoded_record_data;
  ASSERT_TRUE(absl::Base64Unescape(base64_record_data, &decoded_record_data));
  absl::StatusOr<absl::Cord> uncompressed_record_data =
      UncompressWithGzip(decoded_record_data);
  confidentialcompute::AttestationVerificationRecord decoded_record;
  ASSERT_TRUE(decoded_record.ParseFromCord(*uncompressed_record_data));
  EXPECT_THAT(decoded_record, EqualsProto(record));
}

}  // namespace

}  // namespace fcp::client::attestation
