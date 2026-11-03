using System;
using System.IO;
using System.Linq;
using System.Text;
using NEAT.Net.Abstrations;
using NEAT.Net.Abstrations.Enums;
using NEAT.Net.Abstrations.Interfaces;
using NEAT.Net.Operators;
using Xunit;

namespace NEAT.Net.Tests
{
    /// <summary>
    /// Mutation and crossover operators plugged into the evaluator
    /// </summary>
    [Collection(RandomCollection.Name)]
    public class OperatorTests
    {
        private sealed class SizeEvaluator : Evaluator
        {
            public SizeEvaluator(IGenome genome, IInnovationGenerator generator, NeatSettings settings)
                : base(genome, generator, settings)
            { }

            public override float EvaluateGenome(IGenome genome) => genome.Nodes.Count + (genome.Connections.Values.Sum(x => Math.Abs(x.Weight)) / 100);
        }

        private sealed class CountingMutation : IMutation
        {
            public float Probability { get; set; } = 1;

            public int Applied { get; private set; }

            public float GetProbability(EvolutionContext context) => Probability;

            public void Apply(IGenome genome, EvolutionContext context) => Applied++;
        }

        private sealed class RecordingCrossover : ICrossover
        {
            private readonly MultipointCrossover inner = new MultipointCrossover();

            public int Calls { get; private set; }

            public bool AlwaysFitterFirst { get; private set; } = true;

            public IGenome Cross(IGenome fitter, IGenome weaker, EvolutionContext context)
            {
                Calls++;
                AlwaysFitterFirst &= fitter.Fitness >= weaker.Fitness;
                return inner.Cross(fitter, weaker, context);
            }
        }

        internal static string Population(Evaluator evaluator)
        {
            return string.Join("|", evaluator.Genomes.Select(genome =>
            {
                using var stream = new MemoryStream();
                GenomeSerializer.Save(genome, stream);
                return Encoding.UTF8.GetString(stream.ToArray());
            }));
        }

        private static SizeEvaluator Create(int seed, NeatSettings settings = null)
        {
            using var random = new SeededRandom(seed);
            var generator = new InnovationGenerator();
            settings ??= new NeatSettings { PopulationSize = 40, AddNodeRate = 0.2f };
            return new SizeEvaluator(Genome.CreateMinimal(2, 1, generator, settings: settings), generator, settings);
        }

        private static void Run(Evaluator evaluator, int seed, int generations)
        {
            using var random = new SeededRandom(seed);
            for (int i = 0; i < generations; i++) evaluator.Evaluate();
        }

        [Fact]
        public void Evaluator_StartsWithTheMutationsAndCrossoverOfThePaper()
        {
            SizeEvaluator evaluator = Create(40);

            Assert.Collection(evaluator.Mutations,
                              x => Assert.IsType<WeightMutation>(x),
                              x => Assert.IsType<AddConnectionMutation>(x),
                              x => Assert.IsType<AddNodeMutation>(x));
            Assert.IsType<MultipointCrossover>(evaluator.Crossover);
            Assert.Throws<ArgumentNullException>(() => evaluator.Crossover = null);
        }

        [Fact]
        public void AddedMutation_IsAppliedToEveryOffspring()
        {
            SizeEvaluator evaluator = Create(41);
            var counting = new CountingMutation();
            evaluator.Mutations.Add(counting);

            Run(evaluator, 42, 3);

            // every offspring except the unchanged champions is mutated
            Assert.InRange(counting.Applied, 3 * (40 - 8), 3 * 40);
        }

        [Fact]
        public void MutationWithProbabilityZero_LeavesTheRunUnchanged()
        {
            SizeEvaluator plain = Create(43);
            SizeEvaluator withZero = Create(43);
            var never = new CountingMutation { Probability = 0 };
            withZero.Mutations.Insert(0, never);

            Run(plain, 44, 8);
            Run(withZero, 44, 8);

            Assert.Equal(0, never.Applied);
            Assert.Equal(Population(plain), Population(withZero));
        }

        [Fact]
        public void WithoutMutations_OffspringOnlyRecombineTheStartingGenes()
        {
            SizeEvaluator evaluator = Create(45);
            evaluator.Mutations.Clear();
            uint[] innovations = evaluator.Genomes.First().Connections.Keys.ToArray();
            float[] weights = evaluator.Genomes.SelectMany(x => x.Connections.Values).Select(x => x.Weight).Distinct().ToArray();

            Run(evaluator, 46, 5);

            Assert.All(evaluator.Genomes, x => Assert.Equal(innovations, x.Connections.Keys));
            Assert.All(evaluator.Genomes.SelectMany(x => x.Connections.Values), x => Assert.Contains(x.Weight, weights));
        }

        [Fact]
        public void CustomCrossover_IsUsedWithTheFitterParentFirst()
        {
            SizeEvaluator evaluator = Create(47);
            var crossover = new RecordingCrossover();
            evaluator.Crossover = crossover;

            Run(evaluator, 48, 5);

            Assert.True(crossover.Calls > 50);
            Assert.True(crossover.AlwaysFitterFirst);
        }

        [Fact]
        public void GenomeMethods_MatchTheOperators()
        {
            var settings = new NeatSettings();
            var generator = new InnovationGenerator();
            Genome start = Genome.CreateMinimal(3, 2, generator);

            IGenome byMethod = start.Clone();
            IGenome byOperator = start.Clone();
            using (new SeededRandom(49))
            {
                byMethod.Mutate(MutationType.Node, generator, settings);
                byMethod.Mutate(MutationType.Connection, generator, settings);
                byMethod.Mutate(MutationType.AllConnections, generator, settings);
            }

            using (new SeededRandom(49))
            {
                var context = new EvolutionContext(generator, settings, NEATUtils.Random);
                new AddNodeMutation().Apply(byOperator, context);
                new AddConnectionMutation().Apply(byOperator, context);
                new WeightMutation().Apply(byOperator, context);
            }

            Assert.Equal(GenomeJson.Of(byMethod), GenomeJson.Of(byOperator));

            byMethod.Fitness = 2;
            byOperator.Fitness = 1;
            IGenome childByMethod, childByOperator;
            using (new SeededRandom(50))
            {
                childByMethod = byOperator.Crossover(byMethod, settings);
            }

            using (new SeededRandom(50))
            {
                childByOperator = new MultipointCrossover().Cross(byMethod, byOperator, new EvolutionContext(null, settings, NEATUtils.Random));
            }

            Assert.Equal(GenomeJson.Of(childByMethod), GenomeJson.Of(childByOperator));
        }
    }
}
