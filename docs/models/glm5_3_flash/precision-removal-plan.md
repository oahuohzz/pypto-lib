# Arithmetic compensation removal plan

## Current acceptance rules

The user superseded the earlier dual-reference investigation on 2026-10-21:

1. Kernels must not use precision compensation, including high/low products,
   residual corrections or compensated transcendental functions.
2. Validate local operators against their existing CPU golden and thresholds
   before proceeding to independent multi-layer propagation.
3. PyTorch CPU golden is the acceptance reference. Native hybrid comparisons
   are diagnostic only and are not an additional acceptance gate.

Current work removes compensation from the production working tree, including
unused helpers. The original source is preserved locally under
`build_output/glm53_cpu_local_first/baseline/`. Local runs keep the existing
CPU golden and thresholds unchanged initially. The historical investigation
below records earlier candidates, not the current acceptance policy.

## Follow-up validation coverage

The validation baseline is frozen in `build_output/glm53_validation_followup/`.
CPU golden formulas and all acceptance tolerances remain unchanged in this
follow-up. Dense and RMSNorm edits affect owning-file fixtures and CLI case
selection. Forward changes add an exclusive `--end-layer` execution bound and
static guards for unreachable DSA/MoE branches in KDA-only configurations.
The kernel tensor ABI, quantization scheme and arithmetic are unchanged.

### Acceptance contract

| Check | CPU reference and acceptance |
| --- | --- |
| RMSNorm BF16 | Existing atol 1e-4, rtol 1/128 comparator |
| RMSNorm INT8 | At most one-code error within the existing 0.1% budget; inactive rows exactly zero |
| RMSNorm FP32 scale | atol 0, rtol two FP32 epsilons, zero outliers; inactive rows exactly zero |
| INT8 Dense output | Existing ratio_reldiff threshold 0.003 and 1% outlier budget |
| BF16 Dense output | atol=rtol=0.02, zero outliers |
| Forward layer outputs, streams, hidden | atol=rtol=0.02, at most 5% outliers; TP replicas exact |
| Forward cache/state outputs | Existing mapped-row/state checks, zero outliers |

Other operator-specific gates remain in their owning files. Native mixed
references are not used for acceptance. Official-weight coverage remains the
previous first-layer BF16-Dense run, not a full W8A8 official-model validation.

### Added local tests

- Dense INT8 now honors `--case single|ragged|aligned|all`, uses a fixed local
  generator, and includes zero and large-magnitude rows. All three default
  cases pass: task task_20261021_201517_14692332110.
- RMSNorm adds `--fixture rounding`; quantization inputs contain positive and
  negative half-integers and adjacent representable values. Both 5- and
  16-token cases pass, including exact inactive rows:
  task task_20261021_201557_148416018633.
- Seven NPU replays use captured Dense inputs from both shards of the first
  three layers, plus a second-step input. They execute the actual RMSNorm and
  Dense kernels with their existing CPU references and budgets. All pass,
  INT8 input codes match exactly, maximum scale difference is 3.72529e-9,
  and output relative L2 is below 1.9e-7:
  task task_20261021_201427_144875211318.

### Independent DSA plus MoE layer

`--layers 4 --start-layer 3 --steps 2 --tp 2 --ep 2` executes only the fourth
layer, initialized directly from the fixture embedding. Task
`task_20261021_201517_1469273648` fails the overall gate (75.43 s): caches,
recurrent state and final hidden pass, but step-0 MoE output has 5.8670%
outliers and step-1 streams have 5.3417%, both above 5%.

CPU boundary replay on that run's captured intermediate inputs passes all
nine groups under the unchanged independent comparison budgets. Thus this
new case is not a fully passing single-layer acceptance result, even though
local computation passes on identical inputs. It exposes sensitivity within
the attention-to-mHC-to-MoE composition.

### Depth comparisons

The original `--layers 2` run hit a known compiler bounds failure before
execution. Static guards exclude unreachable sparse/MoE branches; compilation
then passes. The corrected two-layer fixture executes but fails its CPU gate,
with 6.6620% step-0 hidden outliers (task task_20261021_201923_16181539457).

Changing `--layers` changes generated weight layouts. For a controlled depth
comparison, retain `--layers 4` and select `--end-layer 1` or `--end-layer 2`.
The CPU reference uses the same execution interval. Unexecuted layer traces
must remain exactly equal to their initial expected values. The one-layer
prefix passes (67.59 s); its first-layer actual and expected traces each
match the prior four-layer run bitwise. Results for the controlled depth
sweep are recorded below.

| Executed prefix, fixed four-layer fixture | CPU result | Step-0 hidden outliers |
| --- | --- | --- |
| One layer, two steps | PASS, all nine output groups | Within existing gate |
| Two layers, two steps | FAIL | 10,106 / 131,072 (7.7103%) |
| Four layers, two steps (prior archived run) | FAIL | 31,204 / 131,072 (23.8068%) |

The two-layer standalone retry completed in 79.75 s, task
`task_20261021_202353_176809920311`. Its first two layers' actual and expected
traces each match the prior four-layer run bitwise, across both decode steps.
The first failing layer boundary is zero-based layer 1 Dense output:
36,780 / 131,072 outliers (28.0609%). Streams and convolution state also fail;
recurrent state passes. Sparse caches remain untouched in this prefix.
Times include compilation and validation and are not performance measurements.

The preceding prefix job reported device-worker failure and stalled before
CPU comparison; it was terminated through the allocator and is not counted
as a numerical result. The standalone retry above produced the numerical
result. No device reset or manual device selection was used.

### Remaining acceptance gaps

The local captured-input checks exclude a reproducible Dense arithmetic error
on those identical inputs; they do not establish independent-chain accuracy.
The DSA-plus-MoE composition still fails intermediate gates, and two- and
four-layer independent propagation still fail. Further fixes must identify
an incorrect operation contract, cast, reduction or state update at the first
diverging boundary, without compensation or relaxed tolerances. If only
permitted floating-point rounding followed by INT8 threshold crossings is
observed, that sensitivity must remain reported as an unresolved acceptance
gap rather than being hidden by a boundary replay or a different reference.

Reproduction scripts, captured tensors, comparator reports, source hashes and
logs are archived in `build_output/glm53_validation_followup/`. These tests
cover synthetic W8A8 fixtures and the previously tested official BF16 first
layer; they do not validate a complete converted official W8A8 checkpoint.

## DSA single-layer localization

The follow-up isolates `--layers 4 --start-layer 3 --steps 2 --tp 2 --ep 2`
using the same captured fixture. No compute arithmetic or CPU golden changes
were made. Reports and temporary composition probes are archived under
`build_output/glm53_dsa_localize/`.

### First differences and propagation

The attention-normalized input matches CPU golden bitwise in both steps.
MLA prolog first introduces small differences: all four step/shard cases pass
the original local gate. Absorption also passes its original gate. Sparse
attention on identical inputs has relative L2 error 0.000833-0.001366; its
original local comparator passes all four cases. The output projection on
identical context inputs has relative L2 error below 1.9e-6 and passes its
original local comparator. Summing the two captured projection partials and
rounding to BF16 reproduces the full-forward attention trace bitwise in both
steps. This validates the diagnostic reconstruction, including the short
fixture's complete selected history; it is not general long-context indexer
coverage.

A CPU diagnostic that reproduces the existing BF16 exponential-weight cast
before PV leaves only 2-7 differing context elements per shard/step, with
relative L2 below 3.8e-5. This supports BF16 PV rounding as the dominant local
attention difference on this fixture. It is diagnostic only: the original
CPU attention golden remains the acceptance reference, and the single BF16
PV implementation is unchanged.

After attention and mHC, 833 and 1,136 of 65,536 MoE input INT8 codes differ
between independent chains in steps 0 and 1. Each differs by one code. CPU
routing selects the same expert sets; routing weights differ slightly.
Replacing only the attention-output boundary in CPU replay with its captured
NPU value removes all forward-tolerance outliers in the downstream MoE output
and streams. Replacing the MoE mixed input has the same effect. This is a
causal diagnostic of propagation, not an independent single-layer pass.

A second CPU intervention keeps the captured MoE input and independently
replaces only the quantized input codes or only the routing weights/indices.
Expert-internal activation quantization remains unchanged:

| CPU intervention | Step-0 MoE outliers / 65,536 | Step-1 MoE outliers / 65,536 |
| --- | ---: | ---: |
| Captured mixed input | 3,837 | 4,782 |
| Replace only input INT8 codes with independent CPU codes | 0 | 0 |
| Replace only routes/weights with independent CPU values | 3,823 | 4,813 |

Counts use the unchanged forward atol=rtol=0.02 against independent CPU output.
Replacing input codes reduces relative L2 from 0.0151/0.0162 to 0.00173/0.00164.
This isolates input-code threshold crossings as the dominant MoE amplification
on this fixture; it does not establish that all quantized workloads behave so.

### Router validation repair

The standalone `moe_gate_test` lacked `pl.Out` on all four result tensors.
Its historical harness PASS did not validate any output and must not be used
as router correctness evidence. The entry now marks all four outputs and
uses deterministic fixtures (seed 533).

The previous index comparator also assumed the returned weights were sorted.
Selection is sorted by sigmoid scores plus correction bias, whereas returned
weights exclude that bias. Validation now requires exactly equal selected
expert multisets and aligns paired weights by expert ID before applying the
unchanged relative-difference threshold 0.003 and 1% outlier budget. A changed
expert set fails. No numerical tolerance was relaxed.

The repaired 64-token default and both captured 16-token inputs pass all four
output comparisons. On captured inputs the deferred-quantization CPU codes
match NPU codes exactly; deferred scale differences are below 7.5e-9. Compared
with the standalone normalized-BF16 reference, 2,902 and 3,178 codes differ by
one and pass its existing one-code tolerance. These are distinct reference
contracts and must not be conflated with upstream propagation differences.

Device tasks:

- MLA prolog, four cases: `task_20261021_204036_248778322857`.
- Absorption, sparse attention and projection: `task_20261021_204137_25311655933`.
- Repaired router default plus captured cases: `task_20261021_204325_262840920972`.

The independent DSA-plus-MoE acceptance failure remains open. Local gates and
boundary substitutions do not supersede it. No compensation or multi-layer
acceptance workaround was introduced.

## Decomposition of attention, mHC and quantization

The next investigation freezes the same DSA single-layer capture. It changes
no model compute source, acceptance golden or numerical threshold. Evidence
is archived in `build_output/glm53_error_decomposition/`.

### mHC and quantization contributions

The production mHC coefficient/post/coefficient/pre composition passes all
original local gates for both steps, task `task_20261021_231200_68813617183`.
Its final mixed input matches the full-forward NPU trace bitwise. The independent
CPU reconstruction likewise matches the archived CPU trace bitwise.

| Contribution | Step 0 | Step 1 |
| --- | ---: | ---: |
| Extra BF16 mixed-input differences from mHC on identical attention input | 2 | 2 |
| Extra INT8 code differences from that mHC error | 0 | 0 |
| Code differences explained by attention input propagated through CPU mHC | 833 | 1,136 |
| Code differences caused by changing activation values only | 833 | 1,136 |
| Code differences caused by changing quantization multipliers only | 0 | 0 |

All 16 per-token maxima and quantization multipliers are identical across
independent chains in each step. Thus input-value perturbations explain the
observed input-code crossings; these are not a changing-amax effect. The
maximum scaled-value perturbation among changed codes is 0.526/0.857, while
the median distance to a half-integer boundary is 0.0575/0.0588. These values
use the production multiplication by `127 / amax`, not division by the rounded
reciprocal scale.

The final dequantization scale also includes inverse RMS and is not identical:
its maximum relative change is about 9.19e-5/8.09e-5. Replacing only that scale
in CPU MoE leaves 3,803/4,798 output outliers (per 65,536 values), compared with
3,837/4,782 without replacement. It is not the main cause on this fixture.

### Contract comparison and rejected numerical shortcut

Source inspection uses the clean local vLLM-Ascend snapshot at
`ff62595360a069343ecdc3b2cd3a0daf45b4f1f0`, not an assertion about upstream latest.

- `vllm_ascend/models/glm5next/model.py:hc_pre` materializes RMSNorm output
  before passing activations to MoE. Dynamic W8A8 quantizes that activation
  in `quantization/methods/base.py:_quant_hidden_states`. GLM here instead
  quantizes the FP32 RMSNorm numerator and folds inverse RMS into scale.
- The local Qwen A8W8 implementation in `decode_layer_a8w8.py` defers inverse
  RMS and rounds the numerator to BF16. The older DeepSeek V4-Flash-MTP gate
  also defers RMSNorm. DeepSeek V4.1 uses explicit BF16 RMSNorm before its
  MXFP8 gate path. These are different quantization contracts, not mutually
  interchangeable numerical references.
- vLLM-Ascend's A3 `hc_post_d_split.h` widens inputs, accumulates the sublayer
  update and residual contributions in FP32, then casts once on output.
  That structure agrees with the current GLM post path; no extra BF16
  accumulation hop was found there.
- A3 NoPE attention routes through `npu_sparse_flash_attention` in
  `attention/sfa_v1.py:sparse_mla`. Its arch22 vector service calls
  `AmlaVecCompute` before narrowing PV inputs: it includes amplitude scaling
  and correction bookkeeping. Therefore a BF16 PV cast alone does not prove
  complete equivalence to that operator. No such correction path was adopted.

A CPU-only alternate-contract experiment quantizes normalized BF16 activations
on BOTH sides, retaining the same weights and captured-versus-independent
mixed inputs. Code differences become 917/1,154, and MoE output outliers are
4,005/4,795 (6.1111%/7.3166%), still above 5%. These are diagnostic results
against a separate paired CPU computation, not acceptance against a changed
golden and not a hardware candidate. They rule out normalization-contract
alignment alone as a sufficient fix for this fixture.

The evidence now points upstream to attention numerical error as the dominant
source of the observed quantization amplification. It does not identify an
incorrect mHC formula, quantization multiplier or expert selection. Remaining
attention changes need a supported ordinary-arithmetic contract and independent
CPU validation. A compensation path, silently changed reference or relaxed gate
would not resolve the stated acceptance requirement. Single-layer and multi-
layer independent acceptance remain open.

## User-authorized amplitude-correction trial

The user subsequently requested an isolated trial of vLLM-Ascend's amplitude
correction. This is a scoped exception for the experiment, not adoption of
arbitrary compensation or a change to CPU acceptance.

Only the isolated candidate's `decode_sparse_attn.py` changes. For each block,
it computes `n = round(-m / ln(2))`, forms the BF16 amplitude
`a = BF16(exp((m / ln(2) + n) * ln(2)))`, and uses one PV matmul with
`BF16(exp(scores-m) * float(a))`. Dividing its FP32 result by `float(a)`
restores the block scale before the existing flash merge. Source cast naming
is followed with `round`. There is no high/low split or second PV.

This adapts `AmlaVecCompute` amplitude selection to independent parallel blocks.
It is not a literal port of the source's sequential accumulator-bit updates,
coefficient approximations or first-block accumulator bias. No bitwise
identity to the complete vLLM-Ascend operator is claimed.

The default multi-block sparse-attention test passes, as do all four captured
context cases under the original local comparator. The TP2 DSA-plus-MoE
single-layer/two-step CPU gate still fails:

| Metric | Baseline | Amplitude candidate |
| --- | ---: | ---: |
| Step-0 attention relative L2 | 0.00230470 | 0.00227183 |
| Step-1 attention relative L2 | 0.00239140 | 0.00237996 |
| Step-0 MoE input code differences | 833 | 845 |
| Step-1 MoE input code differences | 1,136 | 1,116 |
| Step-0 MoE output outliers | 5.8670% | 6.0532% |
| Step-1 MoE output outliers | 7.3517% | 7.1350% |
| Step-1 streams outliers | 5.3417% | 5.0800% |

Hidden and caches pass in both runs. The nine CPU golden output groups are
bitwise unchanged. The candidate attention probes reproduce its full-forward
attention trace bitwise in both steps. Thus the comparison is controlled,
but the slight attention improvement does not reliably improve quantized
composition or meet the 5% gate.

The candidate is preserved in `build_output/glm53_amplitude_trial/` and is not
promoted to the working tree. No performance conclusion is drawn. The native
source's complete sequential implementation remains untested by this trial.

Tasks: captured attention `task_20261021_232352_112822816967`; standalone
`task_20261021_232407_114557829470` (PASS); TP2 layer
`task_20261021_232423_11547185295` (numerical FAIL).

## Local-first execution results

The current working tree contains no arithmetic.py and no compensated BF16
linear helper. Seven former arithmetic consumers use ordinary FP32 operations;
linear projections use ordinary BF16 products with FP32 accumulation. The
existing CPU golden formulas have not changed.

### Completed local coverage

- Linear: FP32 accumulation output and BF16 narrowing.
- RMSNorm: plain, fused INT8 and residual-add cases, both 5 and 16 tokens.
- mHC: coefficients, pre, post and head, both 16 and 20 tokens.
- Dense: INT8 default and BF16 single/ragged cases.
- KDA: projection and output single/ragged cases; recurrence single/batch;
  convolution decode single/ragged and shared prefill ragged.
- Sparse attention; decode indexer step and attention-ABI composition; gate;
  shared and routed experts.
- TP2 all-reduce, distributed MoE and the composed MoE block.

These selected cases pass. The decode-indexer entry needed one stale import
fixed: sylvester_hadamard is owned by indexer_cache, not indexer.

One local comparator changed explicitly: derived RMSNorm FP32 output scales
now allow relative error of two FP32 machine epsilons (approximately 2.384e-7),
zero absolute tolerance and zero outliers. The aligned case had three one-ULP
scale differences, maximum 3.72529e-9, while all INT8 codes passed. The former
bitwise-equality requirement forced agreement with an FP64-derived inverse.
Inactive INT8 rows and scales still require exact zeros. Other local and all
multi-layer tolerances remain unchanged. Default RMSNorm validation now covers
both the ragged and aligned shapes.

### Composed results

| Validation | Result | Task |
| --- | --- | --- |
| Synthetic first-layer TP2, two steps | Pass, 60.03 s | task_20261021_193102_360944113660 |
| Official-weight first-layer TP2, two steps, BF16 Dense | Pass, 25.88 s | task_20261021_193102_36095679139 |
| Synthetic four-layer TP2, two steps, INT8 Dense | Fail, 83.93 s | task_20261021_193335_374052713305 |

The four-layer independent CPU gate has 31,204/131,072 hidden outliers at
step 0 (23.8068%, allowed 5%). The first failing layer-output gate is layer 1
Dense (zero-based), at 28.0609%. Latent, raw, pooled and convolution caches
also fail; recurrent state and pool scales pass.

Replaying the captured intermediate inputs through the CPU golden passes all
nine output groups under the independent comparator budgets, including the
strict cache checks. This is local boundary evidence, not independent
propagation acceptance. The replay uses the same deterministic inputs and
initial states; it does not execute another device kernel.

A CPU sensitivity probe quantizes both the device's recorded Dense input and
the independent golden input using the same CPU quantizer. At step 0/layer 0,
701/65,536 BF16 input values differ, producing 49 INT8 code differences, each
one code. The first activation requantization differs at 423 and 505 values
on the two ranks. By layer 1, 7,953 input codes differ. Thus the probe demonstrates
amplification of upstream differences across quantization boundaries; it does
not identify every upstream rounding source or prove the error acceptable.

An isolated FP32-only CPU-reference experiment increases the step-0 hidden
outlier ratio to 27.5391%. It was rejected; production golden remains unchanged.
The native hybrid reference is not a gate in this work.

An isolated ordinary vector-operation mHC variant passes local mHC tests but
still fails the four-layer CPU gate: step-0 hidden 28,662/131,072 outliers
(21.8674%, allowed 5%), task task_20261021_193806_397401614215. Its small
improvement does not close propagation acceptance; the simpler validated
production implementation is retained. No compensation is reintroduced.

### Reproduction artifacts

Local logs, capture tensors, result JSON, source snapshots and drivers are under
`build_output/glm53_cpu_local_first/`. Local tasks:

- Basic arithmetic/Dense: task_20261021_192803_350635515769.
- KDA cases: task_20261021_192803_350639722190.
- Attention/router/experts: task_20261021_192908_35448257187.
- Repaired indexer entry: task_20261021_193010_358036329969.
- TP2 components: task_20261021_192908_354536420709.
- Aligned mHC and prefill convolution: task_20261021_193127_362879313259.
- Corrected RMSNorm scale contract: task_20261021_193252_37130162271.
- Default six-case RMSNorm entry: task_20261021_194037_405813513871, pass.

The initial indexer import error and exact-scale mismatch are retained in the
result history. They are superseded by the targeted successful reruns.

## Historical investigation

The following sections describe the earlier dual-reference experiments.
Their integration policy is superseded by the current local-first rules above.

## Objective and acceptance

Remove `arithmetic.py` and its compensation from the GLM decode path, using
ordinary operations with explicit dtype, rounding and reduction contracts.
Scope is pypto-lib; serving, embeddings, LM head and MTP remain out of scope.
Keep independent golden references and acceptance thresholds unchanged. A local
boundary pass does not replace the independent four-layer TP2 gate.

## Execution checklist

- [x] Freeze baseline and preserve the working source and existing diagnostics.
  The isolated noarith_convdiv candidate has exact first-layer attention against
  the native hybrid reference. Official weights pass; four-layer gate fails.
- [x] Isolate Dense on identical step-1 inputs: RMSNorm INT8 codes/scales,
  gate/up integer accumulations and dequantization, activation, second INT8
  codes/scales, down projection and TP reduction. Fix the first causal mismatch.
- [x] Isolate mHC coefficient generation: normalization, projection, sigmoid,
  softmax and Sinkhorn. Identical-coefficient HcPost is already exact.
- [ ] Validate combined ordinary-arithmetic candidate; trace any remaining
  independent failure through later KDA, DSA, indexer and MoE boundaries.
- [ ] Integrate validated changes in operator-owned files, remove compensation
  and obsolete paths, preserve colocated tests, and update documentation.
- [ ] Close regression gates: affected operator cases, official-weight first
  layer TP2, independent four-layer TP2/two-step run, and explicit boundary
  validation. Check source imports, lint and diff integrity.

## Evidence rules

Record candidate source identity, reference identity, command/task identifier,
result and limitations. Native hybrid means actual native mHC and native Torch
operations plus CPU integer/DSA/MoE references; it is not a complete vLLM run.
Do not change a reference solely to obtain agreement. Any reference correction
requires a demonstrated operator-contract discrepancy and separate reporting.
Do not replace compensation with another helper or modify precision thresholds.

## Current evidence

See the arithmetic-removal sections in [the model guide](index.md) and local
artifacts under `build_output/glm53_noarith_removal/`. Production source still
uses compensation until the independent replacement is validated.

### Dense isolation evidence

The step-1/rank-0 trace identified a diagnostic-reference bug: `127 / tensor`
in the native hybrid worker used reciprocal multiplication, unlike the explicit
`torch.div(full_like(amax, 127), amax)` in the unchanged production CPU golden.
At source 3.84228515625 and row maximum 7.6845703125, explicit division produces
63.499996185302734 before rounding (code 63); reciprocal multiplication produces
code 64. The kernel agrees with the production CPU quantizer at every code.

After correcting only this diagnostic expression, both quantized code tensors
and integer GEMM accumulators match exactly. Final FP32 local Dense output has
maximum absolute difference 2.39e-6 and zero outliers. Task:
`task_20261021_181703_122273518503`; artifacts `glm_dense_trace_v2.*`.

The corrected hybrid full run still fails: first failing layer-output gate is
now layer 2 Dense; step-0 hidden has 7,640/131,072 outliers. The unchanged CPU
four-layer gate also fails. Neither result permits production removal yet.

The mHC source audit identifies native A3 HcPre split-K size 768 for the
allocated 24-cube-core device (22 partitions including a tail), FP32 cube
RMS statistics, square root followed by division, and ordinary vector
normalization. The candidate previously used four K partitions and a vector
square reduction. Source-aligned ordinary variants are under validation.

## Execution results (2026-10-21)

The isolation and candidate experiments are complete. Production integration
and full regression acceptance remain blocked; compensation has not been
removed. All 37 production model source hashes match the preserved baseline.

### mHC ordinary arithmetic

On the allocated A3 device, reproduce native split-K with 22 partitions of
768 (last partition 256), FP32 cube square statistics, square root followed
by tensor division, and separate vector coefficient operations. Native-input
replay establishes the row reduction `(v0 + v1) + (v2 + v3)` and sequential
column reduction. With those orders, post and residual coefficients match
actual native HcPre exactly on both captured steps, without compensation.

The remaining raw projection differs by at most 7.16e-7 while inverse RMS
matches exactly. Native HcPre explicitly enables HF32 round-toward-zero for
projection; the current PyPTO path uses FP32 and exposes no equivalent HF32
switch found in the DSL audit. This is a concrete contract difference, but
its sole causality has not been established by a controlled HF32 replay.

A separate compiler reproducer shows that two FP32 GEMM accumulators sharing
an operand can corrupt the first output. Separate SPMD scopes avoid this in
the candidate. The minimal reproducer and single-GEMM control are recorded in
the local issue record; no toolchain changes were made.

| Candidate / validation | Result | Task |
| --- | --- | --- |
| mhcnative, four owning mHC cases | All pass | task_20261021_184142_204059023284 |
| mhcnative, official-weight first-layer TP2 | CPU pass; frozen native zero outliers at both steps | task_20261021_184003_195146317973 |
| mhcnative, explicit boundary replay | Pass | task_20261021_184003_195233012414 |
| mhcnative, independent four-layer TP2 | Fail; step-1 hidden 7,734/131,072 outliers | task_20261021_182507_149477214242 |
| mhcpaired, native-input coefficient replay | Exact post and residual | task_20261021_184302_20915046297 |
| mhcpaired, independent four-layer TP2 | Fail; step-1 hidden 7,716/131,072 outliers (5.8868%, allowed 5%) | task_20261021_184354_211825415894 |

The final mhcpaired candidate also fails latent, raw, pooled and convolution
cache gates. Its first failing layer-output gate is layer 2 Dense. Operator,
official-weight and boundary coverage in the table belongs to mhcnative;
it must not be attributed to mhcpaired. Boundary replay uses observed inputs
and cannot substitute for independent propagation.

### Reference acceptance conflict

Regenerating the unchanged production CPU golden and comparing it directly
with the corrected independent native hybrid reference exposes incompatible
cache acceptance intervals. At updated latent-cache index `[0, 0, 6, 15]`:

| Reference | Value | Allowed interval at atol=rtol=0.02 |
| --- | --- | --- |
| Production CPU golden | -0.71875 | [-0.753125, -0.684375] |
| Corrected native hybrid | -0.60546875 | [-0.637578, -0.573359] |

The intervals do not intersect. Cache gates allow zero outliers, so no single
output can pass both at this element. Across the latent tensor, 1,162 elements
have disjoint intervals. The hybrid is not a complete vLLM execution: DSA/MoE
and integer GEMMs use CPU references, recurrent KDA uses Torch operations,
and TP sums are reconstructed. Therefore this conflict does not authorize
silently replacing the production golden with the hybrid.

### Remaining prerequisites

1. Establish one authoritative native operator contract for the independent
   fixture, resolving CPU versus native reference divergence explicitly.
   Keep the mathematical CPU golden as a separate diagnostic if a native
   execution golden is introduced; do not silently rewrite acceptance.
2. Establish equivalent projection semantics for native HF32 versus PyPTO
   FP32, then replay mHC and propagate through the four-layer fixture.
3. Require all independent cache and hidden gates to pass before integrating
   ordinary arithmetic, deleting arithmetic.py and removing obsolete helpers.

The isolated execution path does not call arithmetic.py. An unused compensated
linear helper and its standalone test still exist in the candidate snapshot;
these are not evidence of a fully cleaned production migration.

### Artifacts and integrity

Local evidence is archived under `build_output/glm53_precision_closure/`:
source snapshots for mhcnative and mhcpaired, diagnostic scripts and logs,
reference comparison JSON, native debug headers and SHA256 manifest. Large
captures remain in `/tmp/glm_*/*.pt` or `/tmp/glm_*.pt` as referenced by scripts.
`source_integrity.json` verifies the 37 unchanged production source files.
Native debug capture was synchronized before reading coefficients and its
normal outputs match the unmodified native implementation exactly. Earlier
unsynchronized captures and diagnostic setup failures are not acceptance data.
