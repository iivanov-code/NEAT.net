using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Spiking;
using ANN.Net.Spiking;
using NEAT.Net.Abstrations.Interfaces;
using NEAT.Net.Networks;
using Xunit;
using NeatNodeType = NEAT.Net.Abstrations.Enums.NodeType;

namespace NEAT.Net.Tests
{
    /// <summary>
    /// Genomes become spiking networks, and NEAT evolves spiking networks directly
    /// </summary>
    [Collection(RandomCollection.Name)]
    public class SpikingEvolutionTests
    {
        private static readonly float[][] inputs = { new[] { 0f, 0f }, new[] { 0f, 1f }, new[] { 1f, 0f }, new[] { 1f, 1f } };
        private static readonly float[] targets = { 0f, 1f, 1f, 0f };

        private sealed class SpikingXorEvaluator : SpikingNetworkEvaluator
        {
            public SpikingXorEvaluator(IGenome genome, IInnovationGenerator generator)
                : base(genome, generator, 150)
            { }

            public static float Error(ISpikingNetwork network) => Enumerable.Range(0, 4).Sum(i => Math.Abs(network.Propagate(inputs[i])[0] - targets[i]));

            public static int Correct(ISpikingNetwork network) => Enumerable.Range(0, 4).Count(i => Math.Abs(network.Propagate(inputs[i])[0] - targets[i]) < 0.5f);

            protected override float EvaluateNetwork(ISpikingNetwork network, IGenome genome)
            {
                float error = Error(network);
                return (4 - error) * (4 - error);
            }
        }

        private sealed class CountingEvaluator : Evaluator
        {
            public CountingEvaluator(IGenome genome, IInnovationGenerator generator, ushort size)
                : base(genome, generator, size)
            { }

            public override float EvaluateGenome(IGenome genome) => genome.Connections.Count + genome.Nodes.Count;
        }

        [Fact]
        public void MinimalGenomeBecomesInputsAndOutputs()
        {
            var generator = new InnovationGenerator();
            IGenome genome = Genome.CreateMinimal(3, 2, generator);

            SpikingGraphSettings settings = genome.ToSpikingSettings();

            Assert.Equal(new[] { 3, 2 }, settings.Layers.Select(x => x.NeuronsCount));
            Assert.Equal(new[] { 3, 4 }, settings.Outputs);
            Assert.All(settings.Synapses, s => Assert.InRange(s.Source, 0, 2));

            // bias connections become constant currents
            foreach (IConnectionGene connection in genome.Connections.Values.Where(c => c.Input.Type == NeatNodeType.Bias))
            {
                int output = genome.Nodes.Values.Where(n => n.Type == NeatNodeType.Output).OrderBy(n => n.Innovation).ToList().IndexOf(connection.Output);
                Assert.Equal(connection.Weight, settings.Bias[3 + output]);
            }
        }

        [Fact]
        public void EvolvedGenomesBecomeLayeredSpikingNetworks()
        {
            using var seed = new SeededRandom(7);
            var generator = new InnovationGenerator();
            var evaluator = new CountingEvaluator(Genome.CreateMinimal(3, 2, generator), generator, 60);
            for (int i = 0; i < 25; i++) evaluator.Evaluate();

            foreach (IGenome genome in evaluator.Genomes)
            {
                SpikingGraphSettings settings = genome.ToSpikingSettings();
                int[] layerOf = settings.Layers.SelectMany((layer, i) => Enumerable.Repeat(i, layer.NeuronsCount)).ToArray();

                int synapses = genome.Connections.Values.Count(c => c.Enabled && c.Input.Type != NeatNodeType.Bias);
                Assert.Equal(synapses, settings.Synapses.Count);
                Assert.Equal(genome.Nodes.Values.Count(n => n.Type != NeatNodeType.Bias), layerOf.Length);

                // every connection goes forward, so a spike crosses the whole network in one step
                Assert.All(settings.Synapses, s => Assert.True(layerOf[s.Target] > layerOf[s.Source]));

                float biasSum = genome.Connections.Values.Where(c => c.Enabled && c.Input.Type == NeatNodeType.Bias).Sum(c => c.Weight);
                Assert.Equal(biasSum, settings.Bias.Sum(), 3);

                ISpikingNetwork network = genome.ToSpikingNetwork();
                Assert.Equal(2, network.Propagate(new[] { 0.3f, 0.9f, 0.5f }).Length);
            }
        }

        [Fact]
        public void ActivationGenesChooseTheSpikeFunctions()
        {
            Assert.IsType<IntegrateAndFireActivation>(SpikingPhenotypeSettings.DefaultActivation(ANN.Net.Abstractions.Enums.ActivationTypes.RELU));
            Assert.Equal(0.01f, Assert.IsType<SignedIntegrateAndFireActivation>(SpikingPhenotypeSettings.DefaultActivation(ANN.Net.Abstractions.Enums.ActivationTypes.LeakyRELU)).NegativeSlope);
            Assert.Equal(2, Assert.IsType<SigmoidRateActivation>(SpikingPhenotypeSettings.DefaultActivation(ANN.Net.Abstractions.Enums.ActivationTypes.HyperbolicTangens)).Gain);

            var generator = new InnovationGenerator();
            IGenome genome = Genome.CreateMinimal(2, 1, generator);
            genome.Nodes.Values.Single(n => n.Type == NeatNodeType.Output).Activation = ANN.Net.Abstractions.Enums.ActivationTypes.RELU;

            ISpikingNetwork network = genome.ToSpikingNetwork(new SpikingPhenotypeSettings { InputActivation = () => new PoissonActivation(1) });
            Assert.IsType<PoissonActivation>(network.GetActivation(0));
            Assert.IsType<IntegrateAndFireActivation>(network.GetActivation(1));
        }

        [Fact]
        public void SpikingNetworkEvaluator_SolvesXor()
        {
            using var seed = new SeededRandom(42);
            var generator = new InnovationGenerator();
            var evaluator = new SpikingXorEvaluator(Genome.CreateMinimal(2, 1, generator), generator);

            int generations = 0;
            for (; generations < 300; generations++)
            {
                evaluator.Evaluate();
                if (SpikingXorEvaluator.Correct(evaluator.FittestGenome.ToSpikingNetwork()) == 4) break;
            }

            ISpikingNetwork best = evaluator.FittestGenome.ToSpikingNetwork();
            Assert.True(SpikingXorEvaluator.Correct(best) == 4, $"after {generations} generations: error {SpikingXorEvaluator.Error(best)}");
        }

        [Fact]
        public void LearnedWeightsGoBackIntoTheGenome()
        {
            var generator = new InnovationGenerator();
            IGenome genome = Genome.CreateMinimal(4, 2, generator);
            foreach (IConnectionGene connection in genome.Connections.Values) connection.Weight = 0.6f;

            var settings = new SpikingPhenotypeSettings
            {
                Learn = true,
                Plasticity = new StdpOptions { PotentiationRate = 0.05f, DepressionRate = 0.05f }
            };
            ISpikingNetwork network = genome.ToSpikingNetwork(settings);
            for (int i = 0; i < 5; i++) network.Learn(new[] { 1f, 0.5f, 0f, 0.25f });

            Dictionary<(int, int), float> learned = network.GetSynapses().ToDictionary(s => (s.Source, s.Target), s => s.Weight);
            Assert.Contains(learned.Values, w => Math.Abs(w - 0.6f) > 1e-4f);

            genome.ApplyWeights(network, settings);

            SpikingGraphSettings rebuilt = genome.ToSpikingSettings(settings);
            Assert.All(rebuilt.Synapses, s => Assert.Equal(learned[(s.Source, s.Target)], s.Weight));
            Assert.All(genome.Connections.Values.Where(c => c.Input.Type == NeatNodeType.Bias), c => Assert.Equal(0.6f, c.Weight));
        }
    }
}
