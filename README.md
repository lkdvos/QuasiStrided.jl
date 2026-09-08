# QuasiStrided.jl

A native-Julia dense tensor contraction engine for `StridedView`s
(`StridedViews.jl`): grouped-axis / block-scatter indexing, packing, and
fixed-shape microkernels, in the style of block-scatter-matrix tensor
contraction (BSMTC). This is a first-milestone CPU implementation — see
`STATUS.md` for what is implemented, executed, and unverified, and
`docs/decisions.md` for frozen interfaces.

Status: bootstrap in progress. Usage instructions and API documentation will
be completed at the end of the first implementation milestone (see
`STATUS.md` for current phase).
