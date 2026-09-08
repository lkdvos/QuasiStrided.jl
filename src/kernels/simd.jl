# OWNER: SIMD implementer (Phase 3). See docs/decisions.md and
# Julia-Microkernel-Tile-Interface-Design.md sections 7-9.
#
# Implements: one explicit-SIMD execute_tile! candidate (using SIMD.jl Vec{N,T})
# matching the scalar kernel's packed-format contract from kernel.jl.
