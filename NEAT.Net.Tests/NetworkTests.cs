using System;
using System.IO;
using ANN.Net;
using System.Linq;
using ActivationTypes = ANN.Net.Abstractions.Enums.ActivationTypes;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Optimizers;
using NEAT.Net.Abstrations.Enums;
using NEAT.Net.Abstrations.Interfaces;
using NEAT.Net.Networks;
using Xunit;

namespace NEAT.Net.Tests
{
    [Collection(RandomCollection.Name)]
    public class NetworkTests
    {
        private static readonly PhenotypeSettings Linear = new PhenotypeSettings
        {
            HiddenActivationType = ActivationTypes.Identity,
            OutputActivationType = ActivationTypes.Identity
        };

        private static (Genome Genome, InnovationGenerator Generator) TwoInputsOneOutput(float w0, float w1, float bias)
        {
            var generator = new InnovationGenerator();
            Genome genome = Genome.CreateMinimal(2, 1, generator);
            IConnectionGene[] connections = genome.Connections.Values.OrderBy(x => x.Innovation).ToArray();
            connections[0].Weight = w0;
            connections[1].Weight = w1;
            connections[2].Weight = bias;
            return (genome, generator);
        }

        [Fact]
        public void ToNetwork_ComputesTheWeightedSumOfTheGenome()
        {
            var (genome, _) = TwoInputsOneOutput(2, -3, 0.5f);

            IGraphNetwork network = genome.ToNetwork(Linear);

            Assert.Equal((2 * 1.5f) - (3 * 2) + 0.5f, network.Propagate(1.5f, 2)[0], 4);
            Assert.Equal(0.5f, network.Propagate(0, 0)[0], 4);
        }

        [Fact]
        public void ToNetwork_FollowsHiddenNodesAndSkipsDisabledConnections()
        {
            using var seed = new SeededRandom(7);
            var (genome, generator) = TwoInputsOneOutput(2, -3, 0.5f);

            genome.Mutate(MutationType.Node, generator);
            IGraphNetwork network = genome.ToNetwork(Linear);

            // with linear activations the split keeps the function unchanged (weights 1 and the old weight)
            Assert.Equal((2 * 1.5f) - (3 * 2) + 0.5f, network.Propagate(1.5f, 2)[0], 4);
            Assert.Single(network.HiddenNodeIds);
            Assert.Equal(genome.Connections.Values.Count(x => x.Enabled), network.ConnectionWeights.Count);
        }

        [Fact]
        public void ToNetwork_PrunesHiddenNodesThatDoNotReachAnOutput()
        {
            using var seed = new SeededRandom(8);
            var (genome, generator) = TwoInputsOneOutput(1, 1, 0);
            genome.Mutate(MutationType.Node, generator);

            INodeGene hidden = genome.Nodes.Values.Single(x => x.Type == NodeType.Hidden);
            genome.Connections.Values.Single(x => x.Input == hidden).Disable();

            IGraphNetwork network = genome.ToNetwork(Linear);

            Assert.Empty(network.HiddenNodeIds);
            Assert.Single(network.Propagate(1, 1));
        }

        [Fact]
        public void SavedNetwork_LoadsBackWithTheSameOutputsAndConnectionIds()
        {
            using var seed = new SeededRandom(9);
            var generator = new InnovationGenerator();
            IGenome genome = Genome.CreateMinimal(2, 1, generator);
            for (int i = 0; i < 10; i++)
            {
                genome.Mutate(i % 2 == 0 ? MutationType.Node : MutationType.Connection, generator);
            }

            IGraphNetwork network = genome.ToNetwork();
            using var stream = new MemoryStream();
            NetworkSerializer.Save(network, stream);
            stream.Position = 0;
            IGraphNetwork loaded = NetworkSerializer.Load<IGraphNetwork>(stream);

            Assert.Equal(network.Propagate(0.3f, -0.8f)[0], loaded.Propagate(0.3f, -0.8f)[0]);
            Assert.Equal(network.ConnectionWeights.Keys.OrderBy(x => x), loaded.ConnectionWeights.Keys.OrderBy(x => x));

            // the weights of a loaded network can still be written back into the genome
            genome.ApplyWeights(loaded);
        }

        [Fact]
        public void ApplyWeights_CopiesTrainedWeightsBackIntoTheGenome()
        {
            var (genome, _) = TwoInputsOneOutput(0.1f, 0.1f, 0.1f);
            IGraphNetwork network = genome.ToNetwork(Linear);
            foreach (LearningRateOptimizer optimizer in network.Optimizers)
            {
                optimizer.LearningRate = 0.05f;
            }

            // learn y = x0 - x1
            var random = new Random(1);
            for (int i = 0; i < 2000; i++)
            {
                float x0 = (float)random.NextDouble(), x1 = (float)random.NextDouble();
                network.Propagate(x0, x1);
                network.Backpropagate(x0 - x1);
            }

            genome.ApplyWeights(network);

            IConnectionGene[] connections = genome.Connections.Values.OrderBy(x => x.Innovation).ToArray();
            Assert.Equal(1f, connections[0].Weight, 1);
            Assert.Equal(-1f, connections[1].Weight, 1);
            Assert.Equal(0f, connections[2].Weight, 1);
        }
    }
}
