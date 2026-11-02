using System;
using System.Linq;
using ANN.Net.Abstractions.Interfaces;
using NEAT.Net.Abstrations.Interfaces;
using NEAT.Net.Networks;
using Xunit;

namespace NEAT.Net.Tests
{
    [Collection(RandomCollection.Name)]
    public class EvaluatorTests
    {
        private sealed class CountingEvaluator : Evaluator
        {
            public CountingEvaluator(IGenome genome, IInnovationGenerator generator, ushort size)
                : base(genome, generator, size)
            { }

            // rewards bigger genomes so that the population keeps changing
            public override float EvaluateGenome(IGenome genome) => genome.Connections.Count + genome.Nodes.Count;
        }

        private sealed class XorEvaluator : NetworkEvaluator
        {
            private static readonly float[][] inputs = { new[] { 0f, 0f }, new[] { 0f, 1f }, new[] { 1f, 0f }, new[] { 1f, 1f } };
            private static readonly float[] targets = { 0f, 1f, 1f, 0f };

            public XorEvaluator(IGenome genome, IInnovationGenerator generator)
                : base(genome, generator, 150)
            { }

            public static int Correct(IGraphNetwork network)
            {
                return Enumerable.Range(0, 4).Count(i => Math.Abs(network.Propagate(inputs[i][0], inputs[i][1])[0] - targets[i]) < 0.5f);
            }

            protected override float EvaluateNetwork(IGraphNetwork network, IGenome genome)
            {
                float error = 0;
                for (int i = 0; i < inputs.Length; i++)
                {
                    error += Math.Abs(network.Propagate(inputs[i][0], inputs[i][1])[0] - targets[i]);
                }

                return (4 - error) * (4 - error);
            }
        }

        [Fact]
        public void Evaluate_KeepsThePopulationSize()
        {
            using var seed = new SeededRandom(10);
            var generator = new InnovationGenerator();
            var evaluator = new CountingEvaluator(Genome.CreateMinimal(3, 2, generator), generator, 50);

            for (int i = 0; i < 30; i++)
            {
                evaluator.Evaluate();
                Assert.Equal(50, evaluator.Genomes.Count);
                Assert.Equal(50, evaluator.Species.Sum(x => x.Count));
            }

            Assert.Equal(30u, evaluator.Generation);
            Assert.True(evaluator.FittestGenome.Fitness > 2 * 4 + 5);
        }

        [Fact]
        public void Evaluate_GivesEveryGenomeItsOwnRandomWeights()
        {
            var generator = new InnovationGenerator();
            var evaluator = new CountingEvaluator(Genome.CreateMinimal(3, 2, generator), generator, 20);

            int distinct = evaluator.Genomes.Select(x => x.Connections.Values.First().Weight).Distinct().Count();

            Assert.True(distinct > 10);
        }

        [Fact]
        public void NetworkEvaluator_SolvesXor()
        {
            using var seed = new SeededRandom(42);
            var generator = new InnovationGenerator();
            var evaluator = new XorEvaluator(Genome.CreateMinimal(2, 1, generator), generator);

            for (int i = 0; i < 300; i++)
            {
                evaluator.Evaluate();
                if (XorEvaluator.Correct(evaluator.FittestGenome.ToNetwork()) == 4) break;
            }

            IGraphNetwork best = evaluator.FittestGenome.ToNetwork();
            Assert.Equal(4, XorEvaluator.Correct(best));
            Assert.NotEmpty(best.HiddenNodeIds);
        }
    }
}
