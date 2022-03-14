// iree-translate /usr/local/google/home/ntv/github/iree/iree/test/e2e/linalg_ext_ops/linalg_transform.mlir -iree-input-type=mhlo --iree-mlir-to-vm-bytecode-module --iree-hal-target-backends=dylib-llvm-aot --iree-codegen-use-sandbox-passes  -o /tmp/aaa.vmfb
// iree-check-module /tmp/aaa.vmfb    --driver=dylib --entry_function=f32

func @f32() {
  %lhs = util.unfoldable_constant dense<[
    [15.0, 14.0, 13.0],
    [12.0, 11.0, 10.0],
    [09.0, 08.0, 07.0],
    [06.0, 05.0, 04.0],
    [03.0, 02.0, 01.0]]> : tensor<5x3xf32>
  %rhs = util.unfoldable_constant dense<[
    [15.0, 14.0, 13.0, 12.0, 11.0],
    [10.0, 09.0, 08.0, 07.0, 06.0],
    [05.0, 04.0, 03.0, 02.0, 01.0]]> : tensor<3x5xf32>
  
  %res = "mhlo.dot"(%lhs, %rhs) : (tensor<5x3xf32>, tensor<3x5xf32>) -> tensor<5x5xf32>

  check.expect_almost_eq_const(%res, dense<[
    [430.0, 388.0, 346.0, 304.0, 262.0],
    [340.0, 307.0, 274.0, 241.0, 208.0],
    [250.0, 226.0, 202.0, 178.0, 154.0],
    [160.0, 145.0, 130.0, 115.0, 100.0],
    [70.0, 64.0, 58.0, 52.0, 46.0]]> : tensor<5x5xf32>) : tensor<5x5xf32>
    
  return
}
