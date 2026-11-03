using System;
using System.Globalization;
using System.Linq;
using NEAT.Net;
using NEAT.Net.Abstrations;
using NEAT.Net.Benchmarks;
using NEAT.Net.Networks;

// usage: NEAT.Net.Benchmarks <problem> <configuration> [seeds] [max generations]
// prints one tab-separated line per seed: problem, configuration, seed, solved, generations, best fitness, hidden nodes, species
string problemName = args.Length > 0 ? args[0] : "xor";
string configuration = args.Length > 1 ? args[1] : "sigmoid";
int seeds = args.Length > 2 ? int.Parse(args[2], CultureInfo.InvariantCulture) : 20;
int maxGenerations = args.Length > 3 ? int.Parse(args[3], CultureInfo.InvariantCulture) : 300;

Problem problem = Problem.All.Single(x => string.Equals(x.Name, problemName, StringComparison.Ordinal));

for (int seed = 0; seed < seeds; seed++)
{
#pragma warning disable S2245 // reproducible benchmarks need seeded random numbers, not secure ones
    NEATUtils.Random = new Random(seed);
#pragma warning restore S2245
    NeatSettings settings = Configurations.Create(configuration);
    var generator = new InnovationGenerator();
    var evaluator = new ProblemEvaluator(problem, Genome.CreateMinimal(problem.Inputs[0].Length, 1, generator, settings: settings), generator, settings);

    int generation = 0;
    bool solved = false;
    while (generation < maxGenerations && !solved)
    {
        evaluator.Evaluate();
        generation++;
        solved = problem.IsSolved(evaluator.Outputs(evaluator.FittestGenome.ToNetwork()));
    }

    int hidden = evaluator.FittestGenome.ToNetwork().HiddenNodeIds.Count;
    Console.WriteLine(string.Join('\t', problem.Name, configuration, seed, solved, generation,
                                  evaluator.FittestGenome.Fitness.ToString("F3", CultureInfo.InvariantCulture), hidden, evaluator.Species.Count));
}
