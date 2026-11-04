using System;
using System.Linq;
using System.Threading;
using ANN.Net.Abstractions.Interfaces;
using NEAT.Net.Abstrations.Interfaces;
using NEAT.Net.Networks;
using Xunit;

namespace NEAT.Net.Tests
{
    [Collection(RandomCollection.Name)]
    public class ParallelEvaluationTests
    {
        private static readonly float[][] inputs = { new[] { 0f, 0f }, new[] { 0f, 1f }, new[] { 1f, 0f }, new[] { 1f, 1f } };
        private static readonly float[] targets = { 0f, 1f, 1f, 0f };

        private sealed class XorEvaluator : CompactNetworkEvaluator
        {
            private int running;

            public XorEvaluator(IGenome genome, IInnovationGenerator generator)
                : base(genome, generator, 150)
            { }

            public int MostAtOnce { get; private set; }

            public bool Slow { get; set; }

            protected override float EvaluateNetwork(ICompactNetwork network, IGenome genome)
            {
                int now = Interlocked.Increment(ref running);
                lock (inputs) MostAtOnce = Math.Max(MostAtOnce, now);
                if (Slow) Thread.Sleep(2);

                float error = 0;
                for (int i = 0; i < inputs.Length; i++) error += Math.Abs(network.Propagate(inputs[i][0], inputs[i][1])[0] - targets[i]);

                Interlocked.Decrement(ref running);
                return (4 - error) * (4 - error);
            }
        }

        private sealed class FailingEvaluator : CompactNetworkEvaluator
        {
            public FailingEvaluator(IGenome genome, IInnovationGenerator generator)
                : base(genome, generator, 40)
            { }

            protected override float EvaluateNetwork(ICompactNetwork network, IGenome genome) => throw new InvalidOperationException("broken fitness");
        }

        private static XorEvaluator Run(int threads, int generations)
        {
            using var seed = new SeededRandom(21);
            var generator = new InnovationGenerator();
            var evaluator = new XorEvaluator(Genome.CreateMinimal(2, 1, generator), generator) { MaxDegreeOfParallelism = threads };
            for (int i = 0; i < generations; i++) evaluator.Evaluate();
            return evaluator;
        }

        private static string Describe(IGenome genome) =>
            string.Join(";", genome.Connections.Values.OrderBy(x => x.Innovation).Select(x => $"{x.Innovation}:{x.Input.Innovation}>{x.Output.Innovation}:{BitConverter.SingleToInt32Bits(x.Weight)}:{x.Enabled}"))
            + "|" + string.Join(";", genome.Nodes.Values.OrderBy(x => x.Innovation).Select(x => $"{x.Innovation}:{x.Type}:{x.Activation}"));

        [Fact]
        public void ParallelEvaluation_EvolvesExactlyLikeOneGenomeAtATime()
        {
            XorEvaluator sequential = Run(1, 40), parallel = Run(-1, 40), three = Run(3, 40);

            Assert.Equal(BitConverter.SingleToInt32Bits(sequential.FittestGenome.Fitness), BitConverter.SingleToInt32Bits(parallel.FittestGenome.Fitness));
            Assert.Equal(Describe(sequential.FittestGenome), Describe(parallel.FittestGenome));
            Assert.Equal(sequential.Genomes.Select(Describe), parallel.Genomes.Select(Describe), StringComparer.Ordinal);
            Assert.Equal(sequential.Genomes.Select(Describe), three.Genomes.Select(Describe), StringComparer.Ordinal);
            Assert.Equal(sequential.Species.Count, parallel.Species.Count);
        }

        [Fact]
        public void NetworkEvaluators_UseEveryProcessor()
        {
            using var seed = new SeededRandom(22);
            var generator = new InnovationGenerator();
            var evaluator = new XorEvaluator(Genome.CreateMinimal(2, 1, generator), generator) { Slow = true };
            Assert.Equal(-1, evaluator.MaxDegreeOfParallelism);

            evaluator.Evaluate();
            if (Environment.ProcessorCount > 1) Assert.True(evaluator.MostAtOnce > 1, "the genomes were evaluated one at a time");

            evaluator.MaxDegreeOfParallelism = 1;
            Assert.Throws<ArgumentOutOfRangeException>(() => evaluator.MaxDegreeOfParallelism = 0);
        }

        [Fact]
        public void FailingFitness_ThrowsItsOwnException()
        {
            var generator = new InnovationGenerator();
            var evaluator = new FailingEvaluator(Genome.CreateMinimal(2, 1, generator), generator);

            var exception = Assert.Throws<AggregateException>(() => evaluator.Evaluate());
            Assert.All(exception.InnerExceptions, x => Assert.IsType<InvalidOperationException>(x));
        }
    }
}
