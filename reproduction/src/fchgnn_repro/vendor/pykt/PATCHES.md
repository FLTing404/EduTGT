# Official pyKT model source provenance

AKT and simpleKT are copied from
[pykt-team/pykt-toolkit](https://github.com/pykt-team/pykt-toolkit), commit
`77c3e90fdb807542194b989656ccac10e5d92e12`. The MIT license is retained in
`LICENSE`; upstream and local SHA-256 checksums are recorded in `PROVENANCE.json`.
`upstream.patch` contains the complete changes to the official model files.
The upstream helper `utils.py` is unchanged. Only its imported symbols are
retained; its standalone mask helpers are not used by these two architectures.

The AKT paper authors' original code is [arghosh/AKT](https://github.com/arghosh/AKT).
The selected implementation here is the official pyKT benchmark integration of
AKT, not a claim to rerun the original 2020 experiment protocol. The simpleKT
authors explicitly release their code through
[pyKT](https://pykt.org/simplekt).

The changes to vendored model code are limited to:

1. Allocate attention masks, zero pads, and relative-distance tensors on the
   input tensor device, allowing CPU, CUDA, and Apple MPS; retain input dtype.
2. Add an optional `return_logits=False` forward argument. When true, expose the
   existing final linear output before sigmoid for the project's stable
   `BCEWithLogitsLoss`. The default official probability output is unchanged.

The adapters in `modern_baselines.py` remap project IDs by +1 to reserve 0 for
trailing padding, use one representative concept per interaction (the existing
dataset interface), reconstruct a sequence with its first observed interaction,
and select positions 1 onward as next-response logits. They never read the
`target` tensor as an input. The unknown final response is 0; strict past-only
attention makes it inaccessible to every returned prediction. Event IDs identify
true sequence positions, preserving unscored real events even when the loss mask
has interior gaps. Custom batches without event IDs use a right-padding fallback
that preserves all history through the last scored target. The project loader
provides event IDs and right-pads every chunk.

Defaults for the project experiment are `d_model=64`, `d_ff=128`,
`final_fc_dim=128`, one attention block, four heads, dropout 0.2, shared Q/K,
and non-separated QA embeddings. AKT retains scalar Rasch difficulty and its
L2 coefficient `1e-5`; its second prediction layer is fixed to 256 by upstream.
simpleKT uses its main `qid` variant (vector question difficulty), a second
prediction layer of 128, and 101 sinusoidal positions for 100 next-item targets.
These controlled settings differ from the papers' dataset-specific hyperparameter
searches. They are strong modern baselines, not a claim of SOTA reproduction.

The existing trainer recognizes AKT's `training_objective` and includes its
official difficulty L2. If the caller explicitly selects an external imbalance
loss instead, the trainer's external objective overrides this method, as it does
for all other models; such experiments should disclose omitted AKT L2 or add it
deliberately in the shared trainer. No external imbalance objective is enabled
by these wrappers.
