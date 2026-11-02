using System;
using System.Collections.Generic;
using System.Linq;
using NEAT.Net.Abstrations;
using NEAT.Net.Abstrations.Enums;
using NEAT.Net.Abstrations.Interfaces;

namespace NEAT.Net
{
    /// <summary>
    /// Runs the NEAT generational loop. Override <see cref="EvaluateGenome"/> to score a genome; fitness must not be
    /// negative (negative values are treated as 0 when offspring are shared out between the species)
    /// </summary>
    public abstract class Evaluator : IEvaluator
    {
        private readonly List<IGenome> nextGeneration = new List<IGenome>();

        protected Evaluator(IGenome initialGenome, IInnovationGenerator innovationGenerator, ushort populationSize = DefaultConstants.POPULATION_SIZE)
        {
            if (initialGenome == null) throw new ArgumentNullException(nameof(initialGenome));
            if (innovationGenerator == null) throw new ArgumentNullException(nameof(innovationGenerator));
            if (populationSize == 0) throw new ArgumentOutOfRangeException(nameof(populationSize));

            Genomes = new List<IGenome>(populationSize);
            Species = new List<ISpecie>();

            InnovationGenerator = innovationGenerator;
            PopulationSize = populationSize;

            for (int i = 0; i < PopulationSize; i++)
            {
                IGenome genome = initialGenome.Clone();
                foreach (IConnectionGene connection in genome.Connections.Values)
                {
                    connection.Weight = Random.NextFloat(-DefaultConstants.INITIAL_WEIGHT_RANGE, DefaultConstants.INITIAL_WEIGHT_RANGE);
                }

                Genomes.Add(genome);
            }
        }

        public IGenome FittestGenome { get; protected set; }
        public ICollection<IGenome> Genomes { get; private set; }
        public IInnovationGenerator InnovationGenerator { get; private init; }
        public ushort PopulationSize { get; private set; }
        public ICollection<ISpecie> Species { get; private set; }
        public uint Generation { get; private set; }

        /// <summary>
        /// Genomes closer than this to the representative of a specie belong to that specie
        /// </summary>
        public float CompatibilityThreshold { get; set; } = DefaultConstants.DT;

        protected static Random Random => NEATUtils.Random;

        public abstract float EvaluateGenome(IGenome genome);

        public void Evaluate()
        {
            EvaluateGenomes();
            CategorizeIntoSpecies();
            PutGenomesIntoNextGeneration();
            BreedGenomes();
            Generation++;
        }

        /// <summary>
        /// Assigns a fitness to every genome of the current generation. Override to evaluate in parallel
        /// </summary>
        public virtual void EvaluateGenomes()
        {
            foreach (var genome in Genomes)
            {
                genome.Fitness = EvaluateGenome(genome);
                UpdateFittestGenome(genome);
            }
        }

        /// <summary>
        /// Every genome joins the first specie whose representative (a random member of the previous generation)
        /// is compatible with it, or founds a new specie. Species left without members die out
        /// </summary>
        public void CategorizeIntoSpecies()
        {
            foreach (var specie in Species)
            {
                specie.StartNewGeneration(Random);
            }

            foreach (var genome in Genomes)
            {
                if (!Species.Any(x => x.TryAddGenome(genome, CompatibilityThreshold)))
                {
                    Species.Add(new Specie(genome));
                }
            }

            Species = Species.Where(x => x.Count > 0).ToList();

            foreach (var specie in Species)
            {
                specie.UpdateStagnation();
            }
        }

        /// <summary>
        /// The champion of every specie with more than five members is copied into the next generation unchanged
        /// </summary>
        public void PutGenomesIntoNextGeneration()
        {
            nextGeneration.Clear();

            foreach (var specie in Species.Where(x => x.Count > DefaultConstants.SPECIES_WITH_MORE_THAN))
            {
                nextGeneration.Add(specie.FittestGenome);
            }
        }

        /// <summary>
        /// Fills the rest of the next generation. Every specie gets offspring in proportion to its average fitness
        /// (explicit fitness sharing); stagnant species get none unless they hold the best genome. Parents are taken
        /// from the best members of the specie; 25% of the offspring are mutated copies, the rest come from crossover
        /// </summary>
        public void BreedGenomes()
        {
            var children = new List<IGenome>(PopulationSize);
            int remaining = PopulationSize - nextGeneration.Count;

            if (remaining > 0 && Species.Count > 0)
            {
                IGenome best = Species.Select(x => x.FittestGenome).MaxBy(x => x.Fitness);
                List<ISpecie> eligible = Species.Where(x => x.Stagnation < DefaultConstants.STAGNATION_LIMIT || x.Genomes.Contains(best)).ToList();

                int[] offspring = AllocateOffspring(eligible, remaining);

                for (int i = 0; i < eligible.Count; i++)
                {
                    BreedSpecie(eligible[i], offspring[i], children);
                }
            }

            Genomes = new List<IGenome>(nextGeneration.Concat(children));
            nextGeneration.Clear();
        }

        protected virtual void MutateGenome(IGenome genome)
        {
            if (Random.NextDouble() < DefaultConstants.CHANCE_OF_WEIGHTS_MUTATION)
            {
                genome.Mutate(MutationType.AllConnections, InnovationGenerator);
            }

            if (Random.NextDouble() < DefaultConstants.PROBABILITY_OF_ADDING_NEW_LINK)
            {
                genome.Mutate(MutationType.Connection, InnovationGenerator);
            }

            if (Random.NextDouble() < DefaultConstants.PROBABILITY_OF_ADDING_NEW_NODE)
            {
                genome.Mutate(MutationType.Node, InnovationGenerator);
            }
        }

        protected void UpdateFittestGenome(IGenome genome)
        {
            if (FittestGenome == null || FittestGenome.Fitness < genome.Fitness)
            {
                // a snapshot, so later generations cannot change it
                FittestGenome = genome.Clone();
            }
        }

        private void BreedSpecie(ISpecie specie, int count, List<IGenome> children)
        {
            if (count <= 0) return;

            int parentsCount = Math.Max(1, (int)Math.Ceiling(specie.Count * DefaultConstants.SURVIVAL_THRESHOLD));
            List<IGenome> parents = specie.Genomes.OrderByDescending(x => x.Fitness).Take(parentsCount).ToList();

            for (int i = 0; i < count; i++)
            {
                IGenome parent1 = parents.RandomElement(Random);
                IGenome child;

                if (parents.Count == 1 || Random.NextDouble() < DefaultConstants.PERCENT_OF_DIRECT_OFFSPRING)
                {
                    child = parent1.Clone();
                }
                else
                {
                    IGenome parent2;
                    if (Species.Count > 1 && Random.NextDouble() < DefaultConstants.INTERSPECIES_MATING_RATE)
                    {
                        ISpecie other = Species.Where(x => x != specie).ToList().RandomElement(Random);
                        parent2 = other.Genomes.ToList().RandomElement(Random);
                    }
                    else
                    {
                        do
                        {
                            parent2 = parents.RandomElement(Random);
                        } while (parent2 == parent1);
                    }

                    child = parent1.Crossover(parent2);
                }

                MutateGenome(child);
                children.Add(child);
            }
        }

        /// <summary>
        /// Shares the offspring out in proportion to the average fitness of every specie, so that the counts
        /// add up to exactly <paramref name="total"/>
        /// </summary>
        private static int[] AllocateOffspring(IReadOnlyList<ISpecie> species, int total)
        {
            double[] scores = species.Select(x => Math.Max(0d, x.AverageFitness)).ToArray();
            double sum = scores.Sum();

            if (sum <= 0 || double.IsNaN(sum) || double.IsInfinity(sum))
            {
                scores = Enumerable.Repeat(1d, species.Count).ToArray();
                sum = species.Count;
            }

            double[] exact = scores.Select(x => x / sum * total).ToArray();
            int[] counts = exact.Select(x => (int)Math.Floor(x)).ToArray();
            int left = total - counts.Sum();

            foreach (int index in Enumerable.Range(0, species.Count).OrderByDescending(i => exact[i] - counts[i]).Take(left))
            {
                counts[index]++;
            }

            return counts;
        }
    }
}
