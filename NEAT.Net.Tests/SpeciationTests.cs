using System.Collections.Generic;
using System.Linq;
using NEAT.Net.Abstrations;
using NEAT.Net.Abstrations.Enums;
using NEAT.Net.Abstrations.Interfaces;
using Xunit;
using ActivationTypes = ANN.Net.Abstractions.Enums.ActivationTypes;

namespace NEAT.Net.Tests
{
    /// <summary>
    /// The activation term of the compatibility distance
    /// </summary>
    [Collection(RandomCollection.Name)]
    public class SpeciationTests
    {
        /// <summary>
        /// A genome with 6 hidden nodes (12 nodes in all) and a copy with <paramref name="changed"/> hidden activations changed
        /// </summary>
        private static (IGenome First, IGenome Second) Pair(int changed)
        {
            using var seed = new SeededRandom(80);
            var generator = new InnovationGenerator();
            IGenome first = Genome.CreateMinimal(3, 2, generator);
            for (int i = 0; i < 6; i++) first.Mutate(MutationType.Node, generator);

            IGenome second = first.Clone();
            foreach (INodeGene node in second.Nodes.Values.Where(x => x.Type == NodeType.Hidden).Take(changed))
            {
                node.Activation = ActivationTypes.Gaussian;
            }

            return (first, second);
        }

        [Fact]
        public void ActivationTerm_CountsMatchingNodesWithDifferentActivations()
        {
            var (first, second) = Pair(2);
            Assert.Equal(12, first.Nodes.Count);

            Assert.Equal(0f, first.CompatibilityDistance(second));
            Assert.Equal(2f, first.CompatibilityDistance(second, c4: 1), 5);           // small genomes: Nn = 1
            Assert.Equal(1.5f * 2, first.CompatibilityDistance(second, c4: 1.5f), 5);
            Assert.Equal(first.CompatibilityDistance(second, c4: 1), second.CompatibilityDistance(first, c4: 1), 5);

            var settings = new NeatSettings { ActivationDifferenceCoefficient = 1 };
            Assert.Equal(2f, first.CompatibilityDistance(second, settings), 5);

            settings.SmallGenomeSize = 0;   // normalized by the 12 nodes
            Assert.Equal(2f / 12, first.CompatibilityDistance(second, settings), 5);
        }

        [Fact]
        public void ActivationTerm_TreatsAMissingActivationAsDifferent()
        {
            var (first, second) = Pair(0);
            second.Nodes.Values.First(x => x.Type == NodeType.Output).Activation = null;

            Assert.Equal(1f, first.CompatibilityDistance(second, c4: 1), 5);
        }

        [Fact]
        public void ActivationTerm_SeparatesSpecies()
        {
            var (first, second) = Pair(3);

            Assert.True(new Specie(first).TryAddGenome(second, new NeatSettings()));
            Assert.False(new Specie(first).TryAddGenome(second, new NeatSettings { ActivationDifferenceCoefficient = 1 }));
            Assert.False(new Specie(first).TryAddGenome(second, dt: 3, c4: 1));
        }

        private sealed class SizeEvaluator : Evaluator
        {
            public SizeEvaluator(IGenome genome, IInnovationGenerator generator, NeatSettings settings)
                : base(genome, generator, settings)
            { }

            public override float EvaluateGenome(IGenome genome) => genome.Nodes.Count;
        }

        private static int SpeciesAfterRun(float c4)
        {
            using var seed = new SeededRandom(81);
            var generator = new InnovationGenerator();
            var settings = new NeatSettings
            {
                PopulationSize = 60,
                AddNodeRate = 0.4f,
                ActivationMutationRate = 0.8f,
                ActivationDifferenceCoefficient = c4,
                HiddenActivations = new List<ActivationTypes> { ActivationTypes.Sigmoid, ActivationTypes.Gaussian, ActivationTypes.Sinusoid, ActivationTypes.RELU }
            };
            var evaluator = new SizeEvaluator(Genome.CreateMinimal(2, 1, generator, settings: settings), generator, settings);
            for (int i = 0; i < 12; i++) evaluator.Evaluate();
            return evaluator.Species.Count;
        }

        [Fact]
        public void ActivationTerm_LeadsToMoreSpeciesInARun()
        {
            Assert.True(SpeciesAfterRun(3) > SpeciesAfterRun(0));
        }
    }
}
