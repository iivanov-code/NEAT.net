# NEAT.net
C# implementation of NeuroEvolution of Augmenting Topologies (NEAT) algorithm

<p align = "center">
   <img src = "https://img.shields.io/github/languages/count/xtremista89/NEAT.Net?style=for-the-badge">
   <img src = "https://img.shields.io/github/license/xtremista89/NEAT.Net?style=for-the-badge">
   <img src = "https://img.shields.io/github/repo-size/xtremista89/NEAT.Net?style=for-the-badge">
   <img src = "https://img.shields.io/github/languages/top/xtremista89/NEAT.Net?style=for-the-badge">
   <img src = "https://img.shields.io/github/last-commit/xtremista89/NEAT.Net?style=for-the-badge">
  </p>
<br>

## Building

NEAT.Net builds its networks with [ANN.Net](https://github.com/iivanov-code/ANN.Net), which must be checked
out next to this repository (`../ANN.Net`). Point the `AnnNetRoot` MSBuild property at the ANN.Net folder
if it lives somewhere else:

```
dotnet test NEAT.Net.Tests -p:AnnNetRoot=/path/to/ANN.Net/
```

## Usage

Derive from `CompactNetworkEvaluator` and score the ANN.Net network of every genome. Fitness must not be negative.

```csharp
class XorEvaluator : CompactNetworkEvaluator
{
    public XorEvaluator(IGenome genome, IInnovationGenerator generator)
        : base(genome, generator, populationSize: 150) { }

    protected override float EvaluateNetwork(ICompactNetwork network, IGenome genome)
    {
        float error = 0;
        error += Math.Abs(network.Propagate(0, 0)[0] - 0);
        error += Math.Abs(network.Propagate(0, 1)[0] - 1);
        error += Math.Abs(network.Propagate(1, 0)[0] - 1);
        error += Math.Abs(network.Propagate(1, 1)[0] - 0);
        return (4 - error) * (4 - error);
    }
}

var generator = new InnovationGenerator();
var evaluator = new XorEvaluator(Genome.CreateMinimal(inputsCount: 2, outputsCount: 1, generator), generator);

for (int i = 0; i < 300; i++)
{
    evaluator.Evaluate();   // one generation
}

IGraphNetwork best = evaluator.FittestGenome.ToNetwork();
```

`CompactNetworkEvaluator` compiles every genome straight into an inference-only network
(`genome.ToCompactNetwork()`), without neuron and synapse objects. Derive from `NetworkEvaluator` instead when the
fitness needs the trainable network, for example to train it with backpropagation. For a population of 150 genomes
with about 50 nodes and 120 connections each, scored on 64 cases:

| | `NetworkEvaluator` | `CompactNetworkEvaluator` |
|---|---|---|
| One generation | 86 ms | 32 ms |
| One propagation | 7.7 µs | 2.1 µs |
| Memory per network | 24 KB | 3.2 KB |

Both network evaluators score the genomes on every processor: `EvaluateNetwork` is called from several threads at
once, each call with its own network, so it must not change shared state or use `NEATUtils.Random`. The fitness
values are assigned in population order after all genomes are scored, so evolution goes exactly as when scoring one
genome after the other, on any number of processors. Set `MaxDegreeOfParallelism = 1` to score them one at a time
(the default of a plain `Evaluator`), or another number to limit the threads. 150 genomes scored on 256 cases, 4 cores:

| | One thread | All cores |
|---|---|---|
| `NetworkEvaluator` | 319 ms | 94 ms |
| `CompactNetworkEvaluator` | 54 ms | 17 ms |

### Spiking networks

Derive from `SpikingNetworkEvaluator` to evolve spiking networks directly, with no analog network trained and
converted first:

```csharp
class SpikingXorEvaluator : SpikingNetworkEvaluator
{
    public SpikingXorEvaluator(IGenome genome, IInnovationGenerator generator) : base(genome, generator, 150) { }

    protected override float EvaluateNetwork(ISpikingNetwork network, IGenome genome)
    {
        float error = 0;
        for (int i = 0; i < 4; i++) error += Math.Abs(network.Propagate(inputs[i])[0] - targets[i]);   // firing rates
        return (4 - error) * (4 - error);
    }
}
```

`genome.ToSpikingNetwork(settings)` (used by the evaluator) makes every input an integrate-and-fire neuron fed with the
input value in every time step, and every hidden and output node a spiking neuron whose spike function follows its
activation gene: integrate-and-fire for ReLU, a neuron with negative spikes for leaky ReLU and the identity, the
sigmoid rate neuron for sigmoid and tanh (`SpikingPhenotypeSettings.Activation` chooses others). Nodes are layered by
depth, so every connection goes forward and a spike crosses the network within one time step; bias connections become
a constant input current. The outputs are the firing rates of the output nodes over `TimeSteps` steps (20 by
default). On XOR with 150 genomes, 10 seeds, spiking networks needed a median of 100 generations
(`CompactNetworkEvaluator`: 104) and solved 8 of the 10 seeds within 500 generations (`CompactNetworkEvaluator`: 10).

With `SpikingPhenotypeSettings.Learn` a network can also change its weights with spike-timing-dependent plasticity
while it is evaluated (`network.Learn(input)`), and `genome.ApplyWeights(network, settings)` writes the learned weights
back into the genome (Lamarckian evolution). Every network keeps the state of its neurons, so each evaluation gets
its own and genomes are scored on every processor as with the other evaluators.

### Settings

All parameters of a run live in `NeatSettings`. The defaults follow the NEAT paper (see `DefaultConstants`).
The evaluator keeps the instance it was given, and changes take effect from the next generation:

```csharp
var settings = new NeatSettings
{
    PopulationSize = 300,
    CompatibilityThreshold = 2.5f,
    AddNodeRate = 0.05f,
    AddConnectionRate = 0.1f,
    MaxWeight = 20f
};

var evaluator = new XorEvaluator(Genome.CreateMinimal(2, 1, generator, settings: settings), generator, settings);
evaluator.Evaluate();
evaluator.Settings.AddNodeRate = 0.01f;      // used from the next generation on
```

(`XorEvaluator` then needs a constructor that passes `NeatSettings` on to `base(genome, generator, settings)`.)
Invalid values are rejected with an `ArgumentException` when the evaluator starts and before every generation.

### Activations

Every node gene carries its own activation. New hidden nodes get a random one of `HiddenActivations`, and outputs
one of `OutputActivations` (both only `Sigmoid` by default, as in the paper). With more than one activation in a
list, the activation mutation (rate `ActivationMutationRate`, 0.1) changes them during evolution, and crossover
passes them on from either parent:

```csharp
var settings = new NeatSettings
{
    HiddenActivations = new List<ActivationTypes>
    {
        ActivationTypes.Sigmoid, ActivationTypes.HyperbolicTangens, ActivationTypes.Gaussian, ActivationTypes.Sinusoid, ActivationTypes.RELU
    }
};
```

Assign new lists as above: the lists already hold `Sigmoid`, so `HiddenActivations = { ... }` would add to it.
In the [benchmarks](docs/benchmarks/activation-evolution.md) this list solved XOR in a median of 9 generations
instead of 130, and every circle and sine run instead of half of them. `ActivationDifferenceCoefficient` (c4) adds
differing activations to the compatibility distance; it did not help in the benchmarks and is 0 by default.

### Mutations and crossovers

The evaluator applies every mutation in `evaluator.Mutations` to every offspring, each with its probability, and
combines parents with `evaluator.Crossover`:

| Mutation | Rate setting | Default |
|---|---|---|
| `WeightMutation` | `WeightMutationRate` | 0.8 |
| `AddConnectionMutation` | `AddConnectionRate` | 0.05 |
| `AddNodeMutation` | `AddNodeRate` | 0.03 |
| `ActivationMutation` | `ActivationMutationRate` | 0.1, when a list offers a choice |
| `ToggleEnableMutation` | `ToggleEnableRate` | 0 (off) |
| `RemoveConnectionMutation` | `RemoveConnectionRate` | 0 (off) |
| `RemoveNodeMutation` | `RemoveNodeRate` | 0 (off) |

| Crossover | Matching genes |
|---|---|
| `MultipointCrossover` (default) | weight and activation from a random parent |
| `AveragingCrossover` | average weight of both parents |
| `SinglePointCrossover` | from the fitter parent before a random cut point, from the weaker one after it |
| `ActivationPreservingCrossover` | as multipoint, but nodes keep the fitter parent's activation |
| `WeightedCrossover` | one of several crossovers per child, chosen by weight |

```csharp
evaluator.Crossover = new WeightedCrossover((new MultipointCrossover(), 0.6), (new AveragingCrossover(), 0.4));
evaluator.Mutations.Add(new MyMutation());   // any IMutation
```

All operators keep networks feed-forward. Operators are code and not part of a checkpoint, so configure them
again on the evaluator that loads one.

### Saving and exporting

```csharp

// save the network and load it again later
best.Save(@"C:\models", "xor");                                         // writes C:\models\xor.ann (JSON)
IGraphNetwork loaded = NetworkSerializer.Load<IGraphNetwork>(@"C:\models\xor.ann");
```

To run the network outside .NET (Python, C++, the browser, mobile), export it to ONNX and load it with
ONNX Runtime or any other ONNX tool. The model has one float input `input` of shape `[batch, inputs]` and
one float output `output` of shape `[batch, outputs]`:

```csharp
OnnxExporter.Export(best, @"C:\models\xor.onnx");
```

## Saving genomes and runs

A saved network holds only the finished phenotype. To keep evolving, save genomes or the whole run instead (JSON):

```csharp
// one genome: node genes, connection genes (innovation, weight, enabled) and fitness
GenomeSerializer.Save(evaluator.FittestGenome, @"C:\models\best.genome.json");
Genome genome = GenomeSerializer.Load(@"C:\models\best.genome.json");

// the whole run: settings, population, species, best genome, generation and innovation history
evaluator.SaveCheckpoint(@"C:\runs\xor.checkpoint.json");

var generator = new InnovationGenerator();
var resumed = new XorEvaluator(Genome.CreateMinimal(2, 1, generator), generator);
resumed.LoadCheckpoint(@"C:\runs\xor.checkpoint.json");   // replaces the settings, population and generator state
resumed.Evaluate();                                          // continues with the next generation
```

New mutations of a loaded genome need an innovation generator that knows its innovation numbers, such as the
one restored by `LoadCheckpoint`. To continue with different parameters, change `resumed.Settings` after loading.
The random number sequence is not part of a checkpoint.

`genome.ToNetwork(settings)` turns any genome into an ANN.Net network. Every node uses the activation of its
gene; `PhenotypeSettings` chooses the input activation, the optimizer, and the activation of genes without one
(genomes saved by older versions). The network can also be trained with backpropagation and its weights written
back with `genome.ApplyWeights(network)`.
