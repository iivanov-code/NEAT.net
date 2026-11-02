using System.Linq;
using NEAT.Net.Abstrations;
using NEAT.Net.Abstrations.Enums;
using NEAT.Net.Abstrations.Interfaces;
using NEAT.Net.Networks;
using Xunit;

namespace NEAT.Net.Tests
{
    [Collection(RandomCollection.Name)]
    public class GenomeTests
    {
        [Fact]
        public void InnovationGenerator_ReturnsTheSameNumberForTheSameStructure()
        {
            var generator = new InnovationGenerator();

            uint first = generator.GetConnectionInnovationNumber(1, 2);
            uint other = generator.GetConnectionInnovationNumber(2, 1);

            Assert.Equal(first, generator.GetConnectionInnovationNumber(1, 2));
            Assert.NotEqual(first, other);
            Assert.Equal(generator.GetNodeInnovationNumber(first), generator.GetNodeInnovationNumber(first));
            Assert.NotEqual(generator.GetNodeInnovationNumber(first), generator.NewNodeInnovationNumber());
        }

        [Fact]
        public void CreateMinimal_ConnectsEveryInputAndTheBiasToEveryOutput()
        {
            Genome genome = Genome.CreateMinimal(3, 2, new InnovationGenerator());

            Assert.Equal(3, genome.Nodes.Values.Count(x => x.Type == NodeType.Input));
            Assert.Equal(1, genome.Nodes.Values.Count(x => x.Type == NodeType.Bias));
            Assert.Equal(2, genome.Nodes.Values.Count(x => x.Type == NodeType.Output));
            Assert.Equal(8, genome.Connections.Count);
        }

        [Fact]
        public void Clone_KeepsDisabledGenesAndPointsToItsOwnNodes()
        {
            Genome genome = Genome.CreateMinimal(2, 1, new InnovationGenerator());
            IConnectionGene disabled = genome.Connections.Values.First();
            disabled.Disable();

            IGenome clone = genome.Clone();

            Assert.False(clone.Connections[disabled.Innovation].Enabled);
            foreach (IConnectionGene connection in clone.Connections.Values)
            {
                Assert.Same(clone.Nodes[connection.Input.Innovation], connection.Input);
                Assert.Same(clone.Nodes[connection.Output.Innovation], connection.Output);
            }
        }

        [Fact]
        public void CompatibilityDistance_CountsDisjointExcessAndWeightDifference()
        {
            using var seed = new SeededRandom(1);
            var generator = new InnovationGenerator();
            Genome first = Genome.CreateMinimal(2, 1, generator);
            IGenome second = first.Clone();

            Assert.Equal(0f, first.CompatibilityDistance(second));

            // split a connection in the second genome: one disjoint/excess gene pair appears on each side
            second.Mutate(MutationType.Node, generator);
            foreach (IConnectionGene connection in second.Connections.Values)
            {
                if (first.Connections.TryGetValue(connection.Innovation, out IConnectionGene matching))
                {
                    connection.Weight = matching.Weight + 1;
                }
            }

            // 3 matching genes that differ by 1, two excess genes; N = 1 for small genomes
            Assert.Equal((2 * DefaultConstants.C1) + (DefaultConstants.C3 * 1f), first.CompatibilityDistance(second), 4);
            Assert.Equal(first.CompatibilityDistance(second), second.CompatibilityDistance(first), 4);
        }

        [Fact]
        public void CompatibilityDistance_HandlesGenomesWithoutConnections()
        {
            var empty = new Genome();
            Genome full = Genome.CreateMinimal(2, 1, new InnovationGenerator());

            Assert.Equal(0f, empty.CompatibilityDistance(new Genome()));
            Assert.Equal(3 * DefaultConstants.C1, empty.CompatibilityDistance(full));
        }

        [Fact]
        public void NodeMutation_SplitsAnEnabledConnection()
        {
            using var seed = new SeededRandom(2);
            var generator = new InnovationGenerator();
            Genome genome = Genome.CreateMinimal(1, 1, generator, withBias: false);
            IConnectionGene original = genome.Connections.Values.Single();

            genome.Mutate(MutationType.Node, generator);

            Assert.False(original.Enabled);
            INodeGene hidden = genome.Nodes.Values.Single(x => x.Type == NodeType.Hidden);
            IConnectionGene into = genome.Connections.Values.Single(x => x.Output == hidden);
            IConnectionGene outOf = genome.Connections.Values.Single(x => x.Input == hidden);
            Assert.Equal(1f, into.Weight);
            Assert.Equal(original.Weight, outOf.Weight);

            // the same split in another genome produces the same node and innovation numbers
            Genome other = (Genome)genome.Clone();
            other.Connections[original.Innovation].Enable();
            other.Nodes.Remove(hidden.Innovation);
            other.Connections.Remove(into.Innovation);
            other.Connections.Remove(outOf.Innovation);
            other.Mutate(MutationType.Node, generator);
            Assert.Contains(hidden.Innovation, other.Nodes.Keys);
            Assert.Contains(into.Innovation, other.Connections.Keys);
            Assert.Contains(outOf.Innovation, other.Connections.Keys);
        }

        [Fact]
        public void Mutations_NeverCreateCycles()
        {
            using var seed = new SeededRandom(3);
            var generator = new InnovationGenerator();
            IGenome genome = Genome.CreateMinimal(3, 2, generator);

            for (int i = 0; i < 300; i++)
            {
                genome.Mutate(i % 3 == 0 ? MutationType.Node : MutationType.Connection, generator);
            }

            Assert.True(genome.Nodes.Values.Count(x => x.Type == NodeType.Hidden) > 50);

            // building the network throws on a cycle, so this also proves the genome is feed-forward
            var network = genome.ToNetwork();
            Assert.Equal(2, network.Propagate(1, 0.5f, -1).Length);
        }

        [Fact]
        public void Crossover_InheritsTheTopologyOfTheFitterParent()
        {
            using var seed = new SeededRandom(4);
            var generator = new InnovationGenerator();
            Genome parent = Genome.CreateMinimal(2, 1, generator);
            IGenome fitter = parent.Clone();
            IGenome weaker = parent.Clone();

            fitter.Mutate(MutationType.Node, generator);
            weaker.Mutate(MutationType.Node, generator);
            weaker.Mutate(MutationType.Node, generator);
            fitter.Fitness = 2;
            weaker.Fitness = 1;

            for (int i = 0; i < 20; i++)
            {
                IGenome child = weaker.Crossover(fitter);

                Assert.Equal(fitter.Connections.Keys.OrderBy(x => x), child.Connections.Keys.OrderBy(x => x));
                Assert.Equal(fitter.Nodes.Keys.OrderBy(x => x), child.Nodes.Keys.OrderBy(x => x));
                foreach (IConnectionGene connection in child.Connections.Values)
                {
                    Assert.Same(child.Nodes[connection.Input.Innovation], connection.Input);
                    Assert.Same(child.Nodes[connection.Output.Innovation], connection.Output);
                }
            }
        }

        [Fact]
        public void Crossover_DisablesGenesDisabledInEitherParentWithSeventyFivePercentChance()
        {
            using var seed = new SeededRandom(5);
            Genome first = Genome.CreateMinimal(1, 1, new InnovationGenerator(), withBias: false);
            IGenome second = first.Clone();
            uint innovation = first.Connections.Keys.Single();
            second.Connections[innovation].Disable();
            first.Fitness = second.Fitness = 1;

            int disabled = Enumerable.Range(0, 4000).Count(_ => !first.Crossover(second).Connections[innovation].Enabled);

            Assert.InRange(disabled / 4000.0, 0.72, 0.78);
        }
    }
}
