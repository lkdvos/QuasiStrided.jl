# OWNER: driver implementer (Phase 3). See docs/decisions.md for the frozen
# contract! signature and Julia-Microkernel-Tile-Interface-Design.md section 10.
#
# Implements: contract! axis-list entry point, reusable planning/workspace
# construction, serial output tiling with multiple K panels.
#
# Frozen signature (do not change without a main-process decision-record update):
#
#   contract!(C::StridedView, alpha::Number,
#             A::StridedView, indA::NTuple{NA,Int},
#             B::StridedView, indB::NTuple{NB,Int},
#             beta::Number,
#             indC::NTuple{NC,Int}) where {NA,NB,NC}
#
# Label semantics: indA/indB/indC attach one Int label per axis, in axis order.
# A label present in both indA and indB but absent from indC is a contracted
# (K) axis. A label present in indC and in exactly one of indA/indB is a free
# axis (M if from A, N if from B). Repeated labels within a single one of
# indA/indB/indC (diagonals), and labels present in indC but absent from both
# indA and indB, are out of scope for this milestone and must raise
# ArgumentError. Every axis length for a shared label must match across the
# operands/tensors that share it.
