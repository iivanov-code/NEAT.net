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

Derive from `NetworkEvaluator` and score the ANN.Net network of every genome. Fitness must not be negative.

```csharp
class XorEvaluator : NetworkEvaluator
{
    public XorEvaluator(IGenome genome, IInnovationGenerator generator)
        : base(genome, generator, populationSize: 150) { }

    protected override float EvaluateNetwork(IGraphNetwork network, IGenome genome)
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

`genome.ToNetwork(settings)` turns any genome into an ANN.Net network (`PhenotypeSettings` chooses the
activations and optimizer). The network can also be trained with backpropagation and its weights written back
with `genome.ApplyWeights(network)`.
