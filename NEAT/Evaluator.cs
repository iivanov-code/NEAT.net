using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.ExceptionServices;
using System.Threading.Tasks;
using NEAT.Net.Abstrations;
using NEAT.Net.Abstrations.Enums;
using NEAT.Net.Abstrations.Interfaces;
using NEAT.Net.Operators;
using NEAT.Net.Serialization;

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
            : this(initialGenome, innovationGenerator, new NeatSettings { PopulationSize = populationSize })
        { }

        /// <summary>
        /// Starts a run with the given parameters. The evaluator keeps using this instance, so changes to it take
        /// effect from the next generation
        /// </summary>
        protected Evaluator(IGenome initialGenome, IInnovationGenerator innovationGenerator, NeatSettings settings)
        {
            if (initialGenome == null) throw new ArgumentNullException(nameof(initialGenome));
            if (innovationGenerator == null) throw new ArgumentNullException(nameof(innovationGenerator));
            ArgumentNullException.ThrowIfNull(settings);
            settings.Validate();

            Settings = settings;
            Genomes = new List<IGenome>(settings.PopulationSize);
            Species = new List<ISpecie>();
            InnovationGenerator = innovationGenerator;

            for (int i = 0; i < PopulationSize; i++)
            {
                IGenome genome = initialGenome.Clone();
                foreach (IConnectionGene connection in genome.Connections.Values)
                {
                    connection.Weight = Random.NextFloat(-settings.InitialWeightRange, settings.InitialWeightRange);
                }

                Genomes.Add(genome);
            }
        }

        public NeatSettings Settings { get; private set; }

        /// <summary>
        /// Mutations applied to every offspring, in order, each with its own probability. Starts with the mutations of
        /// the NEAT paper (weights, new connection, new node) followed by activation, toggle and removal mutations,
        /// whose default probability is 0 until their rates are set (activations: until there is a choice)
        /// </summary>
        public IList<IMutation> Mutations { get; } = new List<IMutation>
        {
            new WeightMutation(),
            new AddConnectionMutation(),
            new AddNodeMutation(),
            new ActivationMutation(),
            new ToggleEnableMutation(),
            new RemoveConnectionMutation(),
            new RemoveNodeMutation()
        };

        /// <summary>
        /// Combines two parents into a crossover child; <see cref="MultipointCrossover"/> by default
        /// </summary>
        public ICrossover Crossover
        {
            get;
            set => field = value ?? throw new ArgumentNullException(nameof(value));
        } = new MultipointCrossover();

        /// <summary>
        /// Context the operators of the current generation work with
        /// </summary>
        protected EvolutionContext Context => new EvolutionContext(InnovationGenerator, Settings, Random);
        public IGenome FittestGenome { get; protected set; }
        public ICollection<IGenome> Genomes { get; private set; }
        public IInnovationGenerator InnovationGenerator { get; private set; }
        public ushort PopulationSize => Settings.PopulationSize;
        public ICollection<ISpecie> Species { get; private set; }
        public uint Generation { get; private set; }

        /// <summary>
        /// Genomes closer than this to the representative of a specie belong to that specie
        /// (shortcut for <see cref="NeatSettings.CompatibilityThreshold"/>)
        /// </summary>
        public float CompatibilityThreshold
        {
            get => Settings.CompatibilityThreshold;
            set => Settings.CompatibilityThreshold = value;
        }

        protected static Random Random => NEATUtils.Random;

        public abstract float EvaluateGenome(IGenome genome);

        /// <summary>
        /// Saves the state of the run (settings, population, species, best genome, generation and innovation history) so that
        /// <see cref="LoadCheckpoint(string)"/> can continue it later. Requires the <see cref="NEAT.Net.InnovationGenerator"/>
        /// </summary>
        public void SaveCheckpoint(string filePath)
        {
            ArgumentException.ThrowIfNullOrEmpty(filePath);

            using FileStream stream = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.None);
            SaveCheckpoint(stream);
        }

        /// <inheritdoc cref="SaveCheckpoint(string)"/>
        public void SaveCheckpoint(Stream stream)
        {
            if (InnovationGenerator is not InnovationGenerator generator)
                throw new NotSupportedException($"Checkpoints need the {nameof(NEAT.Net.InnovationGenerator)} to save the innovation history");

            CheckpointSerializer.Save(stream, new CheckpointState(Generation, Settings, Settings.CompatibilityThreshold, generator, Genomes, Species, FittestGenome));
        }

        /// <summary>
        /// Replaces the state of this evaluator, including <see cref="Settings"/>, with a checkpoint saved by
        /// <see cref="SaveCheckpoint(string)"/>. To continue with different parameters (a larger population, for
        /// example), change <see cref="Settings"/> after loading. The random number sequence is not part of the checkpoint
        /// </summary>
        public void LoadCheckpoint(string filePath)
        {
            ArgumentException.ThrowIfNullOrEmpty(filePath);

            using FileStream stream = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read);
            LoadCheckpoint(stream);
        }

        /// <inheritdoc cref="LoadCheckpoint(string)"/>
        public void LoadCheckpoint(Stream stream)
        {
            CheckpointState state = CheckpointSerializer.Load(stream);

            Generation = state.Generation;
            if (state.Settings != null)
            {
                Settings = state.Settings;
            }
            else
            {
                Settings.CompatibilityThreshold = state.CompatibilityThreshold;
            }

            InnovationGenerator = state.InnovationGenerator;
            Genomes = state.Genomes;
            Species = state.Species;
            FittestGenome = state.FittestGenome;
            nextGeneration.Clear();
        }

        public void Evaluate()
        {
            Settings.Validate();

            EvaluateGenomes();
            CategorizeIntoSpecies();
            PutGenomesIntoNextGeneration();
            BreedGenomes();
            Generation++;
        }

        /// <summary>
        /// Most genomes evaluated at the same time: 1 evaluates them one after the other, -1 uses every processor.
        /// With more than one, <see cref="EvaluateGenome"/> is called from several threads at once and must not change
        /// shared state or use <see cref="NEATUtils.Random"/>. The fitness values are assigned in population order once
        /// all genomes are evaluated, so evolution goes exactly as when evaluating one genome after the other
        /// </summary>
        public int MaxDegreeOfParallelism
        {
            get;
            set => field = value == -1 || value >= 1
                ? value
                : throw new ArgumentOutOfRangeException(nameof(value), value, "Must be -1 or at least 1");
        } = 1;

        /// <summary>
        /// Assigns a fitness to every genome of the current generation, on up to <see cref="MaxDegreeOfParallelism"/>
        /// threads
        /// </summary>
        public virtual void EvaluateGenomes()
        {
            if (MaxDegreeOfParallelism == 1)
            {
                foreach (var genome in Genomes)
                {
                    genome.Fitness = EvaluateGenome(genome);
                    UpdateFittestGenome(genome);
                }

                return;
            }

            IGenome[] genomes = Genomes.ToArray();
            var fitness = new float[genomes.Length];
            try
            {
                Parallel.For(0, genomes.Length, new ParallelOptions { MaxDegreeOfParallelism = MaxDegreeOfParallelism },
                             i => fitness[i] = EvaluateGenome(genomes[i]));
            }
            catch (AggregateException exception) when (exception.InnerExceptions.Count == 1)
            {
                ExceptionDispatchInfo.Capture(exception.InnerExceptions[0]).Throw();
            }

            for (int i = 0; i < genomes.Length; i++)
            {
                genomes[i].Fitness = fitness[i];
                UpdateFittestGenome(genomes[i]);
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
                AssignToSpecie(genome);
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

            foreach (var specie in Species.Where(x => x.Count > Settings.ElitismMinimumSpecieSize))
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
                List<ISpecie> eligible = Species.Where(x => x.Stagnation < Settings.StagnationLimit || x.Genomes.Contains(best)).ToList();

                int[] offspring = AllocateOffspring(eligible, remaining);

                for (int i = 0; i < eligible.Count; i++)
                {
                    BreedSpecie(eligible[i], offspring[i], children);
                }
            }

            Genomes = new List<IGenome>(nextGeneration.Concat(children));
            nextGeneration.Clear();
        }

        /// <summary>
        /// Applies every mutation of <see cref="Mutations"/> with its probability. Mutations with probability 0 are
        /// skipped without drawing a random number
        /// </summary>
        protected virtual void MutateGenome(IGenome genome)
        {
            EvolutionContext context = Context;

            foreach (IMutation mutation in Mutations)
            {
                float probability = mutation.GetProbability(context);
                if (probability > 0 && Random.NextDouble() < probability)
                {
                    mutation.Apply(genome, context);
                }
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

        private void AssignToSpecie(IGenome genome)
        {
            if (!Species.Any(x => x.TryAddGenome(genome, Settings)))
            {
                Species.Add(new Specie(genome));
            }
        }

        private void BreedSpecie(ISpecie specie, int count, List<IGenome> children)
        {
            if (count <= 0) return;

            int parentsCount = Math.Max(1, (int)Math.Ceiling(specie.Count * Settings.SurvivalThreshold));
            List<IGenome> parents = specie.Genomes.OrderByDescending(x => x.Fitness).Take(parentsCount).ToList();

            for (int i = 0; i < count; i++)
            {
                IGenome parent1 = parents.RandomElement(Random);
                IGenome child;

                if (parents.Count == 1 || Random.NextDouble() < Settings.MutationOnlyRate)
                {
                    child = parent1.Clone();
                }
                else
                {
                    IGenome parent2;
                    if (Species.Count > 1 && Random.NextDouble() < Settings.InterspeciesMatingRate)
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

                    var (fitter, weaker) = GenomeOperations.OrderParents(parent1, parent2, Random);
                    child = Crossover.Cross(fitter, weaker, Context);
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
