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
out next to this repository (`../ANN.Net`). Point the `AnnNetProject` MSBuild property at
`ANN.Net/NeuralNetwork/ANN.Net.csproj` if it lives somewhere else:

```
dotnet test NEAT.Net.Tests -p:AnnNetProject=/path/to/ANN.Net/NeuralNetwork/ANN.Net.csproj
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

`genome.ToNetwork(settings)` turns any genome into an ANN.Net network (`PhenotypeSettings` chooses the
activations and optimizer). The network can also be trained with backpropagation and its weights written back
with `genome.ApplyWeights(network)`.
