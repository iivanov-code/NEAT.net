# Plan: evolving activation functions, new mutations and crossovers

Branch: `feature/activation-evolution`

## Goal

Every node gene carries its own activation function, which evolution can change, inherit through
crossover and use when measuring how different two genomes are. On top of that, mutation and
crossover become pluggable operators, so new ones can be added and mixed without editing `Genome`.

Today all hidden nodes share one activation chosen in `PhenotypeSettings.HiddenActivationType`
(`NEAT/Networks/PhenotypeSettings.cs:13`), and mutation/crossover are hard-coded in
`Genome` (`NEAT/Genome.cs:159`, `:221`) and `Evaluator.MutateGenome` (`NEAT/Evaluator.cs:193`).

The ANN.Net side is mostly ready: `GraphNode.ActivationType` already overrides the activation per
node, and the JSON serializer and ONNX exporter already handle per-node activations.

## Phase 0 – Prerequisites in ANN.Net (separate branch in ANN.Net) – done

Done on ANN.Net `feature/activation-evolution`. Besides Gaussian and Swish it also fixed:
- the SQNL derivative, which was non-zero where the function is flat (|x| > 2)
- GELU, which had no parameterless constructor, so its shared instance threw
- ISRU, which had no shared instance

14 activations can now be built, saved and exported to ONNX.
BReLU stays excluded (decision: exclude).

Only seven activations can be built today (`NetworkUtils.GetActivation`, `NeuralNetwork/Utils/NetworkUtils.cs:139`):
Sigmoid, TanH, ArcTan, ReLU, LeakyReLU, BinaryStep and Identity. The other activation classes exist
but are not wired up, and two of them are wrong:

| Activation | Problem | Fix |
|---|---|---|
| `GaussianActivation` | computes `e^(+x²)` | `e^(−x²)` and its derivative `−2x·e^(−x²)` |
| `SwishActivation` | computes `x / (1 + eˣ − x)` | `x / (1 + e^(−x))` and its derivative |
| `BReLUActivation` | keeps a call counter, so the result depends on call order | leave out of evolution |

Steps:
1. Fix Gaussian and Swish, with derivative tests (gradient check like the transformer tests).
2. Map Gaussian, Sinusoid, SoftSign, SQNL, ISRU, GELU and Swish in `GetActivation` and `GetActivationType`.
3. Map them in the ONNX exporter (`NeuralNetwork/Onnx/OnnxExporter.cs:248`):
   - Sinusoid → `Sin`
   - SoftSign → `Softsign`
   - Gaussian → `Exp(Neg(Mul(x, x)))`
   - Swish → `Mul(x, Sigmoid(x))`
   - GELU → the tanh formula ANN.Net uses (not opset-20 `Gelu`, whose default is the erf form)
   - ISRU and SQNL from elementary operators.
4. Extend the ONNX tests and the save/load tests with every newly mapped activation.

These are the activations that make mixed-activation NEAT useful. Gaussian and Sinusoid in particular
let small networks represent symmetric and repeating patterns, as in CPPNs/HyperNEAT.

## Phase 1 – Configurable settings (NEAT.net) – done

- `NeatSettings` lives in NEAT.Abstractions, which now references ANN.Net.Abstractions; both sibling paths
  come from the `AnnNetRoot` MSBuild property.
- `Evaluator`, `NetworkEvaluator`, `Genome` (mutation, crossover, distance, `CreateMinimal`) and `Specie`
  take the settings; the old signatures use the defaults.
- Checkpoints store the settings, and older checkpoints without them still load.
- Default settings reproduce the previous commit exactly: best fitness and species count matched over
  5 seeds × 40 XOR generations.

The activation set and its probabilities must be configurable per run, but all parameters are
`const` in `DefaultConstants`. Introduce a `NeatSettings` class:

- Copy every `DefaultConstants` value as a property, with the current value as default.
- Add a new activation section:
  - `HiddenActivations` (allowed set, default `{ Sigmoid }` so behaviour is unchanged)
  - `OutputActivations` (default `{ Sigmoid }`)
  - `ActivationMutationRate`
  - `CompatibilityActivationCoefficient` (c4)
- Pass it to `Evaluator` (new constructor overload) and down to the mutation and crossover operators.
- Store it in checkpoints, so a resumed run uses the same parameters.

`DefaultConstants` stays for compatibility and supplies the defaults.

## Phase 2 – Activation in the genome

1. **Gene:** add `ActivationTypes Activation { get; set; }` to `INodeGene` / `NodeGene`
   (`NEAT.Abstrations/Interfaces/INodeGene.cs`, `NEAT/NodeGene.cs`).
   `Clone` copies it. Input and bias nodes keep `Identity`.
   - Recommendation: use ANN.Net's `ActivationTypes` by letting NEAT.Abstractions reference the small
     ANN.Net.Abstractions project. A separate NEAT enum would need a mapping that can drift out of date.
     If NEAT.Abstractions must stay free of ANN.Net, add a NEAT `ActivationFunction` enum plus one
     mapping table in `NEAT/Networks`.
2. **New nodes:** `Genome.CreateMinimal` gives outputs an activation from `OutputActivations`.
   `AddNodeMutation` (`NEAT/Genome.cs:361`) gives the new hidden node a random member of
   `HiddenActivations`.
   - Alternative to evaluate: inherit the activation of the split connection's target node.
     It keeps the network closer to its parent, in line with the "split is almost neutral" idea.
3. **Phenotype:** `ToGraphSettings` (`NEAT/Networks/GenomeNetworkExtensions.cs:31`) passes the
   gene's activation as `GraphNode.ActivationType`. `PhenotypeSettings` activations become the
   fallback only.
4. **Serialization:** `NodeGeneDocument` gets an optional `activation`.
   - Files without it load with the old fixed activation, so old genome files and checkpoints still
     load. Bump the format version to 2 and keep reading version 1.
5. **Historical markings stay as they are:** an activation is a node attribute, like a weight is for
   a connection, so it does not get its own innovation number.

## Phase 3 – Pluggable operators

```csharp
public interface IMutation
{
    float Probability { get; }                                   // per offspring
    void Apply(IGenome genome, MutationContext context);         // generator, settings, random
}

public interface ICrossover
{
    IGenome Cross(IGenome fitter, IGenome weaker, CrossoverContext context);
}
```

- `Evaluator` gets `IList<IMutation> Mutations` and `ICrossover Crossover`, filled with today's
  behaviour by default, so existing runs do not change.
- `MutateGenome` becomes a loop over `Mutations`.
- `Genome.Mutate(MutationType, …)` and `Genome.Crossover` stay as thin wrappers over the default
  operators, for compatibility.
- The existing alignment of genes (`AlignGenes`) moves to a shared internal helper the crossovers use.

## Phase 4 – New mutations

| Mutation | What it does | Default rate |
|---|---|---|
| `ActivationMutation` | replaces a random hidden node's activation with another from `HiddenActivations` (outputs only if `OutputActivations` has more than one entry) | 0.1 |
| `ToggleEnableMutation` | re-enables a disabled connection if that does not create a cycle, or disables an enabled one whose target keeps another input | 0.01 |
| `RemoveConnectionMutation` | removes a connection gene (complexity control, as in neat-python) | 0 (opt-in) |
| `RemoveNodeMutation` | removes a hidden node and its connections | 0 (opt-in) |
| `WeightReplaceMutation` / `WeightPerturbMutation` | today's weight mutation split into two operators with their own rates | as today |

Every structural mutation keeps the feed-forward guarantee: the existing reachability check
(`IsReachable`) is reused.

## Phase 5 – Crossovers

The standard crossover gains the activation rule: a matching node gene inherits its activation
randomly from either parent, just as matching connection genes inherit weights.

- Today the child copies all node genes from the fitter parent (`NEAT/Genome.cs:182`), so this
  needs node alignment, the same as for connections.
- Nodes that only the fitter parent has keep that parent's activation.

New crossover operators (original NEAT C++ has the first three):

| Crossover | Matching genes |
|---|---|
| `MultipointCrossover` (today's) | weight and activation from a random parent |
| `AveragingCrossover` | weight is the average of both parents; activation from a random parent |
| `SinglePointCrossover` | genes up to a random innovation come from one parent, the rest from the other |
| `ActivationPreservingCrossover` | weights as multipoint, but every node keeps the fitter parent's activation (for problems where activation churn hurts) |

`Evaluator` can pick among several crossovers with weights, for example 60% multipoint and 40% averaging.

## Phase 6 – Speciation

Add an activation term to the compatibility distance (`NEAT/Genome.cs:123`):

```
δ = c1·E/N + c2·D/N + c3·W̄ + c4·A/Nn
```

`A` is the number of matching node genes with different activations, and `Nn` is the node-gene count
of the larger genome (1 for small genomes).

- c4 = 0 (the default) gives today's behaviour.
- Start tuning at c4 ≈ 1, so that a different activation weighs like a disjoint gene.
- `ISpecie.TryAddGenome` and `IGenome.CompatibilityDistance` get the new coefficient as an optional
  parameter (default 0).

## Phase 7 – Tests and benchmarks

**Unit tests**
- Every operator.
- Cycle-freedom after random mixes of all structural mutations (extends `Mutations_NeverCreateCycles`).
- Activation inheritance rates in crossover.
- The distance term.
- Version-1 files still load.
- Checkpoint determinism with the new operators (extends the existing resume test).

**Integration**
- A genome with mixed activations gives the same output in ANN.Net, after the JSON round trip and in
  ONNX Runtime.

**Benchmarks** (20 seeds each, report solved rate and mean generations, as was done for XOR)
- XOR: must not get worse with the default settings.
- Circle classification (inside/outside a circle): Gaussian nodes should help.
- Sine regression or two spirals: Sinusoid nodes should help.
- Compare fixed Sigmoid against `{ Sigmoid, Tanh, Gaussian, Sinusoid, ReLU }`.

## Order of work and size

| Phase | Repo | Size | Depends on |
|---|---|---|---|
| 0 ANN.Net activations | ANN.Net | M | – |
| 1 NeatSettings | NEAT.net | M | – |
| 2 Activation gene | NEAT.net | M | 1 (and 0 for the extra activations) |
| 3 Operator interfaces | NEAT.net | M | 1 |
| 4 New mutations | NEAT.net | M | 2, 3 |
| 5 Crossovers | NEAT.net | M | 2, 3 |
| 6 Speciation term | NEAT.net | S | 2 |
| 7 Benchmarks | both | S | all |

Phases 0 and 1 can run in parallel, and each phase is a separate commit that keeps all tests green.
Defaults reproduce today's behaviour until the benchmarks justify changing them.

## Breaking changes

- `INodeGene` gains `Activation`.
- `IGenome.CompatibilityDistance` and `ISpecie.TryAddGenome` gain an optional coefficient.
  This is source-compatible, but not binary-compatible.
- The genome and checkpoint format moves to version 2, and version 1 still loads.

## Decisions

Recommendations accepted:
1. Output activations stay fixed unless `OutputActivations` lists more than one function.
2. NEAT.Abstractions references ANN.Net.Abstractions and uses its `ActivationTypes`.
3. Per-node response and bias are not part of this plan.
4. BReLU is excluded from evolution.

## Open questions (resolved, see Decisions)

1. Should output activations evolve too, or stay fixed per problem? Proposed: fixed unless
   `OutputActivations` lists more than one function.
2. NEAT.Abstractions referencing ANN.Net.Abstractions, or a NEAT-specific enum with a mapping?
3. Should evolution also cover a per-node response (slope) and bias, as in neat-python? It would
   bring back the paper's steepened sigmoid without a new activation. Not in this plan; it fits the
   same gene extension later.
4. Should `BReLU` be fixed or excluded? It is stateful and cannot be reproduced in ONNX.
