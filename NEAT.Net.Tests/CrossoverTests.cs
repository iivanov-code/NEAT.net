using System;
using System.Collections.Generic;
using System.Linq;
using NEAT.Net.Abstrations;
using NEAT.Net.Abstrations.Enums;
using NEAT.Net.Abstrations.Interfaces;
using NEAT.Net.Operators;
using Xunit;
using ActivationTypes = ANN.Net.Abstractions.Enums.ActivationTypes;

namespace NEAT.Net.Tests
{
    [Collection(RandomCollection.Name)]
    public class CrossoverTests
    {
        private static NeatSettings Mixed() => new NeatSettings
        {
            HiddenActivations = new List<ActivationTypes> { ActivationTypes.Sigmoid, ActivationTypes.Gaussian, ActivationTypes.Sinusoid }
        };

        /// <summary>
        /// Two parents from one ancestor: the weaker has every matching weight +1 and every hidden activation changed,
        /// the fitter one has an extra node
        /// </summary>
        private static (IGenome Fitter, IGenome Weaker) Parents()
        {
            using var seed = new SeededRandom(70);
            var generator = new InnovationGenerator();
            NeatSettings settings = Mixed();
            IGenome ancestor = Genome.CreateMinimal(3, 2, generator, settings: settings);
            for (int i = 0; i < 6; i++) ancestor.Mutate(MutationType.Node, generator, settings);

            IGenome fitter = ancestor.Clone();
            IGenome weaker = ancestor.Clone();
            fitter.Mutate(MutationType.Node, generator, settings);

            foreach (IConnectionGene connection in weaker.Connections.Values) connection.Weight += 1;
            foreach (INodeGene node in weaker.Nodes.Values.Where(x => x.Type == NodeType.Hidden))
            {
                node.Activation = node.Activation == ActivationTypes.Gaussian ? ActivationTypes.Sinusoid : ActivationTypes.Gaussian;
            }

            fitter.Fitness = 2;
            weaker.Fitness = 1;
            return (fitter, weaker);
        }

        private static IGenome Cross(ICrossover crossover, IGenome fitter, IGenome weaker)
        {
            return crossover.Cross(fitter, weaker, new EvolutionContext(null, Mixed(), NEATUtils.Random));
        }

        /// <summary>
        /// Inherited weights are copies, so they are compared exactly
        /// </summary>
        private static bool Same(float a, float b) => BitConverter.SingleToInt32Bits(a) == BitConverter.SingleToInt32Bits(b);

        private static List<uint> Matching(IGenome fitter, IGenome weaker) => fitter.Connections.Keys.Where(weaker.Connections.ContainsKey).OrderBy(x => x).ToList();

        private static void AssertStructureOfFitter(IGenome child, IGenome fitter)
        {
            Assert.Equal(fitter.Nodes.Keys.OrderBy(x => x), child.Nodes.Keys.OrderBy(x => x));
            Assert.Equal(fitter.Connections.Keys.OrderBy(x => x), child.Connections.Keys.OrderBy(x => x));
            MutationTests.AssertValid(child);
        }

        [Fact]
        public void Multipoint_InheritsWeightsAndActivationsFromBothParents()
        {
            var (fitter, weaker) = Parents();
            using var seed = new SeededRandom(71);
            List<INodeGene> sharedHidden = fitter.Nodes.Values.Where(x => x.Type == NodeType.Hidden && weaker.Nodes.ContainsKey(x.Innovation)).ToList();
            int fromWeakerActivation = 0, fromWeakerWeight = 0, total = 0;

            for (int i = 0; i < 400; i++)
            {
                IGenome child = Cross(new MultipointCrossover(), fitter, weaker);
                AssertStructureOfFitter(child, fitter);

                foreach (INodeGene node in sharedHidden)
                {
                    ActivationTypes? activation = child.Nodes[node.Innovation].Activation;
                    Assert.True(activation == node.Activation || activation == weaker.Nodes[node.Innovation].Activation);
                    if (activation != node.Activation) fromWeakerActivation++;
                }

                foreach (uint innovation in Matching(fitter, weaker))
                {
                    float weight = child.Connections[innovation].Weight;
                    Assert.True(Same(weight, fitter.Connections[innovation].Weight) || Same(weight, weaker.Connections[innovation].Weight));
                    if (Same(weight, weaker.Connections[innovation].Weight)) fromWeakerWeight++;
                    total++;
                }
            }

            Assert.InRange(fromWeakerActivation / (400.0 * sharedHidden.Count), 0.45, 0.55);
            Assert.InRange(fromWeakerWeight / (double)total, 0.45, 0.55);
        }

        [Fact]
        public void Averaging_AveragesMatchingWeights()
        {
            var (fitter, weaker) = Parents();
            using var seed = new SeededRandom(72);

            IGenome child = Cross(new AveragingCrossover(), fitter, weaker);

            AssertStructureOfFitter(child, fitter);
            foreach (uint innovation in Matching(fitter, weaker))
            {
                Assert.Equal(fitter.Connections[innovation].Weight + 0.5f, child.Connections[innovation].Weight, 5);
            }

            foreach (uint innovation in fitter.Connections.Keys.Except(weaker.Connections.Keys))
            {
                Assert.Equal(fitter.Connections[innovation].Weight, child.Connections[innovation].Weight);
            }
        }

        [Fact]
        public void SinglePoint_TakesTheFitterParentBeforeTheCutAndTheWeakerFromIt()
        {
            var (fitter, weaker) = Parents();
            using var seed = new SeededRandom(73);
            List<uint> matching = Matching(fitter, weaker);
            var cuts = new HashSet<int>();

            for (int i = 0; i < 100; i++)
            {
                IGenome child = Cross(new SinglePointCrossover(), fitter, weaker);
                AssertStructureOfFitter(child, fitter);

                bool[] fromWeaker = matching.Select(x => Same(child.Connections[x].Weight, weaker.Connections[x].Weight)).ToArray();
                int cut = Array.IndexOf(fromWeaker, true);
                Assert.True(cut >= 0, "the gene at the cut point comes from the weaker parent");
                Assert.All(fromWeaker.Take(cut), Assert.False);
                Assert.All(fromWeaker.Skip(cut), Assert.True);
                cuts.Add(cut);
            }

            Assert.True(cuts.Count > matching.Count / 2, "the cut point is random");
        }

        [Fact]
        public void ActivationPreserving_KeepsTheActivationsOfTheFitterParent()
        {
            var (fitter, weaker) = Parents();
            using var seed = new SeededRandom(74);

            for (int i = 0; i < 50; i++)
            {
                IGenome child = Cross(new ActivationPreservingCrossover(), fitter, weaker);
                AssertStructureOfFitter(child, fitter);
                Assert.Equal(fitter.Nodes.Values.Select(x => (x.Innovation, x.Activation)), child.Nodes.Values.Select(x => (x.Innovation, x.Activation)));
            }
        }

        [Fact]
        public void Weighted_PicksCrossoversInProportionToTheirWeights()
        {
            var (fitter, weaker) = Parents();
            using var seed = new SeededRandom(75);
            var weighted = new WeightedCrossover((new MultipointCrossover(), 3), (new AveragingCrossover(), 1));
            uint probe = Matching(fitter, weaker)[0];

            int averaged = Enumerable.Range(0, 2000)
                                     .Count(_ => Math.Abs(Cross(weighted, fitter, weaker).Connections[probe].Weight - (fitter.Connections[probe].Weight + 0.5f)) < 1e-5);

            Assert.InRange(averaged / 2000.0, 0.22, 0.28);
            Assert.Throws<ArgumentException>(() => new WeightedCrossover());
            Assert.Throws<ArgumentException>(() => new WeightedCrossover((new MultipointCrossover(), 0)));
            Assert.Throws<ArgumentException>(() => new WeightedCrossover((null, 1)));
        }

        private sealed class SizeEvaluator : Evaluator
        {
            public SizeEvaluator(IGenome genome, IInnovationGenerator generator, NeatSettings settings)
                : base(genome, generator, settings)
            { }

            public override float EvaluateGenome(IGenome genome) => genome.Nodes.Count + (genome.Connections.Values.Sum(x => Math.Abs(x.Weight)) / 100);
        }

        public static TheoryData<string> CrossoverNames() => new TheoryData<string> { "multipoint", "averaging", "single-point", "activation-preserving", "weighted" };

        [Theory]
        [MemberData(nameof(CrossoverNames))]
        public void EveryCrossover_KeepsAValidPopulation(string name)
        {
            ICrossover crossover = name switch
            {
                "multipoint" => new MultipointCrossover(),
                "averaging" => new AveragingCrossover(),
                "single-point" => new SinglePointCrossover(),
                "activation-preserving" => new ActivationPreservingCrossover(),
                _ => new WeightedCrossover((new MultipointCrossover(), 1), (new AveragingCrossover(), 1), (new SinglePointCrossover(), 1))
            };

            using var seed = new SeededRandom(76);
            var generator = new InnovationGenerator();
            NeatSettings settings = Mixed();
            settings.PopulationSize = 40;
            settings.AddNodeRate = 0.3f;
            settings.ToggleEnableRate = 0.1f;
            var evaluator = new SizeEvaluator(Genome.CreateMinimal(2, 1, generator, settings: settings), generator, settings) { Crossover = crossover };

            for (int i = 0; i < 15; i++) evaluator.Evaluate();

            Assert.Equal(40, evaluator.Genomes.Count);
            Assert.All(evaluator.Genomes, MutationTests.AssertValid);
        }
    }
}
