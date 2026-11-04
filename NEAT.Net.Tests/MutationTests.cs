using System;
using System.Collections.Generic;
using System.Linq;
using NEAT.Net.Abstrations;
using NEAT.Net.Abstrations.Enums;
using NEAT.Net.Abstrations.Interfaces;
using NEAT.Net.Networks;
using NEAT.Net.Operators;
using Xunit;
using ActivationTypes = ANN.Net.Abstractions.Enums.ActivationTypes;

namespace NEAT.Net.Tests
{
    /// <summary>
    /// Activation, toggle and removal mutations
    /// </summary>
    [Collection(RandomCollection.Name)]
    public class MutationTests
    {
        private static NeatSettings Mixed() => new NeatSettings
        {
            HiddenActivations = new List<ActivationTypes> { ActivationTypes.Sigmoid, ActivationTypes.Gaussian, ActivationTypes.Sinusoid },
            ToggleEnableRate = 0.2f,
            RemoveConnectionRate = 0.1f,
            RemoveNodeRate = 0.05f
        };

        private static EvolutionContext Context(InnovationGenerator generator, NeatSettings settings) => new EvolutionContext(generator, settings, NEATUtils.Random);

        private static IGenome WithHiddenNodes(InnovationGenerator generator, NeatSettings settings, int splits)
        {
            IGenome genome = Genome.CreateMinimal(2, 2, generator, settings: settings);
            for (int i = 0; i < splits; i++) genome.Mutate(MutationType.Node, generator, settings);
            return genome;
        }

        /// <summary>
        /// Every connection points to nodes of the genome, and the graph of all connections (disabled ones too) has no cycle
        /// </summary>
        internal static void AssertValid(IGenome genome)
        {
            foreach (IConnectionGene connection in genome.Connections.Values)
            {
                Assert.Same(genome.Nodes[connection.Input.Innovation], connection.Input);
                Assert.Same(genome.Nodes[connection.Output.Innovation], connection.Output);
            }

            var inDegree = genome.Nodes.Keys.ToDictionary(x => x, _ => 0);
            foreach (IConnectionGene connection in genome.Connections.Values) inDegree[connection.Output.Innovation]++;
            var queue = new Queue<uint>(inDegree.Where(x => x.Value == 0).Select(x => x.Key));
            int visited = 0;
            while (queue.Count > 0)
            {
                uint node = queue.Dequeue();
                visited++;
                foreach (IConnectionGene connection in genome.Connections.Values.Where(x => x.Input.Innovation == node))
                {
                    if (--inDegree[connection.Output.Innovation] == 0) queue.Enqueue(connection.Output.Innovation);
                }
            }

            Assert.Equal(genome.Nodes.Count, visited);
        }

        [Fact]
        public void ActivationMutation_IsOffWithoutAChoice()
        {
            Assert.Equal(0f, new ActivationMutation().GetProbability(Context(new InnovationGenerator(), new NeatSettings())));
            Assert.Equal(0.1f, new ActivationMutation().GetProbability(Context(new InnovationGenerator(), Mixed())));

            // duplicates are no choice either
            var duplicates = new NeatSettings { HiddenActivations = new List<ActivationTypes> { ActivationTypes.Gaussian, ActivationTypes.Gaussian } };
            Assert.Equal(0f, new ActivationMutation().GetProbability(Context(new InnovationGenerator(), duplicates)));
        }

        [Fact]
        public void ActivationMutation_ChangesOneHiddenNodeToAnotherAllowedActivation()
        {
            using var seed = new SeededRandom(60);
            var generator = new InnovationGenerator();
            NeatSettings settings = Mixed();
            IGenome genome = WithHiddenNodes(generator, settings, 6);

            for (int i = 0; i < 50; i++)
            {
                var before = genome.Nodes.Values.ToDictionary(x => x.Innovation, x => x.Activation);
                genome.Mutate(MutationType.Activation, generator, settings);

                var changed = genome.Nodes.Values.Where(x => x.Activation != before[x.Innovation]).ToList();
                INodeGene node = Assert.Single(changed);
                Assert.Equal(NodeType.Hidden, node.Type);
                Assert.Contains(node.Activation.Value, settings.HiddenActivations);
            }
        }

        [Fact]
        public void ActivationMutation_ChangesOutputsOnlyWhenTheyHaveAChoice()
        {
            using var seed = new SeededRandom(61);
            var generator = new InnovationGenerator();
            var settings = new NeatSettings { OutputActivations = new List<ActivationTypes> { ActivationTypes.Sigmoid, ActivationTypes.Identity } };
            IGenome genome = Genome.CreateMinimal(2, 2, generator, settings: settings);
            genome.Mutate(MutationType.Node, generator, settings);

            for (int i = 0; i < 30; i++) genome.Mutate(MutationType.Activation, generator, settings);

            Assert.All(genome.Nodes.Values.Where(x => x.Type == NodeType.Hidden), x => Assert.Equal(ActivationTypes.Sigmoid, x.Activation));
            Assert.Contains(genome.Nodes.Values, x => x.Type == NodeType.Output && x.Activation == ActivationTypes.Identity);
        }

        [Fact]
        public void ToggleEnable_EnablesDisabledConnectionsAndKeepsLastInputs()
        {
            var generator = new InnovationGenerator();
            Genome genome = Genome.CreateMinimal(1, 1, generator, withBias: false);
            IConnectionGene only = genome.Connections.Values.Single();
            var toggle = new ToggleEnableMutation();

            toggle.Apply(genome, Context(generator, new NeatSettings()));
            Assert.True(only.Enabled); // the only input of the output stays enabled

            only.Disable();
            toggle.Apply(genome, Context(generator, new NeatSettings()));
            Assert.True(only.Enabled);

            Genome two = Genome.CreateMinimal(2, 1, generator, withBias: false);
            using var seed = new SeededRandom(62);
            toggle.Apply(two, Context(generator, new NeatSettings()));
            Assert.Single(two.Connections.Values, x => !x.Enabled);
        }

        [Fact]
        public void RemoveConnection_RemovesOneGene()
        {
            using var seed = new SeededRandom(63);
            var generator = new InnovationGenerator();
            IGenome genome = WithHiddenNodes(generator, new NeatSettings(), 3);
            int count = genome.Connections.Count;

            genome.Mutate(MutationType.RemoveConnection, generator);

            Assert.Equal(count - 1, genome.Connections.Count);
            AssertValid(genome);
        }

        [Fact]
        public void RemoveNode_RemovesAHiddenNodeAndItsConnections()
        {
            using var seed = new SeededRandom(64);
            var generator = new InnovationGenerator();
            IGenome genome = WithHiddenNodes(generator, new NeatSettings(), 3);
            var hiddenBefore = genome.Nodes.Values.Where(x => x.Type == NodeType.Hidden).Select(x => x.Innovation).ToList();

            genome.Mutate(MutationType.RemoveNode, generator);

            var hiddenAfter = genome.Nodes.Values.Where(x => x.Type == NodeType.Hidden).Select(x => x.Innovation).ToList();
            uint removed = Assert.Single(hiddenBefore.Except(hiddenAfter));
            Assert.DoesNotContain(genome.Connections.Values, x => x.Input.Innovation == removed || x.Output.Innovation == removed);
            AssertValid(genome);

            Genome plain = Genome.CreateMinimal(2, 1, generator);
            plain.Mutate(MutationType.RemoveNode, generator);
            Assert.Equal(4, plain.Nodes.Count);
        }

        [Fact]
        public void RandomMixOfAllMutations_KeepsGenomesValidAndFeedForward()
        {
            using var seed = new SeededRandom(65);
            var generator = new InnovationGenerator();
            NeatSettings settings = Mixed();
            IGenome genome = Genome.CreateMinimal(3, 2, generator, settings: settings);
            MutationType[] types = Enum.GetValues<MutationType>();

            for (int i = 0; i < 600; i++)
            {
                genome.Mutate(types[NEATUtils.Random.Next(types.Length)], generator, settings);
                if (i % 50 == 0) AssertValid(genome);
            }

            AssertValid(genome);
            Assert.Equal(2, genome.ToNetwork().Propagate(0.5f, -1f, 2f).Length);
        }

        private sealed class SizeEvaluator : Evaluator
        {
            public SizeEvaluator(IGenome genome, IInnovationGenerator generator, NeatSettings settings)
                : base(genome, generator, settings)
            { }

            public override float EvaluateGenome(IGenome genome)
            {
                return genome.Nodes.Values.Count(x => x.Activation == ActivationTypes.Gaussian) + (genome.Connections.Count / 10f);
            }
        }

        [Fact]
        public void Evaluator_WithEveryMutationOn_KeepsAValidPopulation()
        {
            using var seed = new SeededRandom(66);
            var generator = new InnovationGenerator();
            NeatSettings settings = Mixed();
            settings.PopulationSize = 40;
            settings.AddNodeRate = 0.3f;
            settings.AddConnectionRate = 0.3f;
            settings.ActivationMutationRate = 0.5f;
            var evaluator = new SizeEvaluator(Genome.CreateMinimal(2, 1, generator, settings: settings), generator, settings);

            for (int i = 0; i < 25; i++) evaluator.Evaluate();

            Assert.Equal(40, evaluator.Genomes.Count);
            Assert.All(evaluator.Genomes, genome =>
            {
                AssertValid(genome);
                Assert.Single(genome.ToNetwork().Propagate(0.1f, 0.9f));
            });

            // fitness rewards Gaussian nodes, so the activation mutation must have spread them
            Assert.True(evaluator.FittestGenome.Nodes.Values.Count(x => x.Activation == ActivationTypes.Gaussian) >= 2);
        }
    }
}
