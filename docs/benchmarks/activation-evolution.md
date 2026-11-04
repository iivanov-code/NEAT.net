# Benchmark: evolving activation functions

Does letting every hidden node evolve its own activation help NEAT? Three problems, three configurations,
20 seeds each (seeds 0–19), at most 300 generations, population 150, all other settings at their defaults.

## Problems

| Problem | Inputs | Cases | Solved when |
|---|---|---|---|
| XOR | 2 | 4 | every case on the right side of 0.5 |
| Circle | 2 | 7 × 7 grid in [-1, 1]², target 1 inside radius 0.6 | 95% of the cases on the right side of 0.5 |
| Sine | 1 | 25 points of [-π, π], target (sin x + 1) / 2 | mean absolute error below 0.05 |

Fitness is (n − Σ|error|)², as in the XOR experiment of the NEAT paper. Outputs use the sigmoid in every
configuration.

## Configurations

| Name | Hidden activations | c4 |
|---|---|---|
| `sigmoid` | Sigmoid (the default) | 0 |
| `mixed` | Sigmoid, Tanh, Gaussian, Sinusoid, ReLU | 0 |
| `mixed-c4` | same as `mixed` | 1 |

`mixed` runs use the default activation mutation rate (0.1) and the default `MultipointCrossover`, which
inherits a node's activation from a random parent.

## Results

| Problem | Configuration | Solved | Mean generations (solved runs) | Median generations (solved runs) | Mean hidden nodes of the best network | Mean species at the end |
|---|---|---|---|---|---|---|
| XOR | sigmoid | 20/20 | 125 | 130 | 3.8 | 38 |
| XOR | mixed | 20/20 | 12 | 9 | 1.7 | 17 |
| XOR | mixed-c4 | 20/20 | 12 | 10 | 1.6 | 28 |
| Circle | sigmoid | 9/20 | 131 | 125 | 3.6 | 3 |
| Circle | mixed | 20/20 | 94 | 84 | 5.2 | 12 |
| Circle | mixed-c4 | 20/20 | 152 | 144 | 6.5 | 23 |
| Sine | sigmoid | 10/20 | 195 | 192 | 5.2 | 5 |
| Sine | mixed | 20/20 | 19 | 16 | 1.7 | 20 |
| Sine | mixed-c4 | 18/20 | 48 | 34 | 3.3 | 33 |

Hidden nodes and species are averaged over all 20 runs, including runs that did not solve the problem.

## Findings

- **Mixed activations win clearly.** They solve every run of every problem, about 10 times faster on XOR and
  sine, and solve circle in all 20 runs where sigmoid-only solves 9. The solutions are also smaller.
- **Part of the gain comes from the right building block.** One Gaussian node solves XOR (it peaks where
  both inputs are equal) and the circle (it peaks at the centre), and one Sinusoid node solves sine. Problems
  without such a match should gain less.
- **The speciation term does not help here.** With c4 = 1, runs keep more species but solve circle and sine
  more slowly. Matching nodes with different activations are already in different species through their
  weights, and splitting species further thins out the population. Keep c4 at 0 unless a problem shows
  otherwise.
- **The defaults are unchanged.** The default settings still reproduce the paper's sigmoid-only NEAT exactly.
  To use mixed activations, set `NeatSettings.HiddenActivations`.

## Reproducing

```
dotnet build NEAT.Net.Benchmarks -c Debug -p:Optimize=true
NEAT.Net.Benchmarks/bin/Debug/net10.0/NEAT.Net.Benchmarks <xor|circle|sine> <sigmoid|mixed|mixed-c4> [seeds] [max generations]
```

Each run prints one tab-separated line per seed: problem, configuration, seed, solved, generations, best
fitness, hidden nodes, species. The benchmarks are built as optimized Debug because ANN.Net does not compile
in Release yet (`DrawingUtils` uses `ISynapse.Input`/`Output`, which exist only in Debug builds).
