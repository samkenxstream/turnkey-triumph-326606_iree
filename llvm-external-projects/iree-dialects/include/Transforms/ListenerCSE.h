//===-- ListenerCSE.h - Common subexpr elimination with a listener --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IREE_SANDBOX_TRANSFORMS_LISTENERCSE_H
#define LLVM_IREE_SANDBOX_TRANSFORMS_LISTENERCSE_H

#include "Transforms/Listener.h"

namespace mlir {
class DominanceInfo;
class Operation;

LogicalResult eliminateCommonSubexpressions(Operation *op,
                                            DominanceInfo *domInfo,
                                            RewriteListener *listener);
} // namespace mlir

#endif // LLVM_IREE_SANDBOX_TRANSFORMS_LISTENERCSE_H
