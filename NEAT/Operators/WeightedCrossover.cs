using System;
using System.Collections.Generic;
using System.Linq;
using NEAT.Net.Abstrations;
using NEAT.Net.Abstrations.Interfaces;

namespace NEAT.Net.Operators
{
    /// <summary>
    /// Uses one of several crossovers for every child, chosen at random in proportion to the weights, for example
    /// 60% <see cref="MultipointCrossover"/> and 40% <see cref="AveragingCrossover"/>
    /// </summary>
    public sealed class WeightedCrossover : ICrossover
    {
        private readonly (ICrossover Crossover, double Weight)[] crossovers;
        private readonly double total;

        public WeightedCrossover(params (ICrossover Crossover, double Weight)[] crossovers)
        {
            ArgumentNullException.ThrowIfNull(crossovers);
            if (crossovers.Length == 0) throw new ArgumentException("At least one crossover is required", nameof(crossovers));
            if (crossovers.Any(x => x.Crossover == null)) throw new ArgumentException("Crossovers must not be null", nameof(crossovers));
            if (crossovers.Any(x => !(x.Weight > 0) || double.IsInfinity(x.Weight))) throw new ArgumentException("Weights must be positive", nameof(crossovers));

            this.crossovers = crossovers.ToArray();
            total = crossovers.Sum(x => x.Weight);
        }

        public IReadOnlyList<(ICrossover Crossover, double Weight)> Crossovers => crossovers;

        public IGenome Cross(IGenome fitter, IGenome weaker, EvolutionContext context)
        {
            if (crossovers.Length == 1) return crossovers[0].Crossover.Cross(fitter, weaker, context);

            double pick = context.Random.NextDouble() * total;
            foreach (var (crossover, weight) in crossovers)
            {
                if (pick < weight) return crossover.Cross(fitter, weaker, context);
                pick -= weight;
            }

            return crossovers[^1].Crossover.Cross(fitter, weaker, context);
        }
    }
}
