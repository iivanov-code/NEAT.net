using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json.Nodes;
using ANN.Net;
using ANN.Net.Abstractions.Interfaces;
using Microsoft.ML.OnnxRuntime;
using NEAT.Net.Abstrations;
using NEAT.Net.Abstrations.Enums;
using NEAT.Net.Abstrations.Interfaces;
using NEAT.Net.Networks;
using Xunit;
using ActivationTypes = ANN.Net.Abstractions.Enums.ActivationTypes;

namespace NEAT.Net.Tests
{
    /// <summary>
    /// Node genes carry their own activation
    /// </summary>
    [Collection(RandomCollection.Name)]
    public class ActivationGeneTests
    {
        private static readonly ActivationTypes[] Mixed = { ActivationTypes.Gaussian, ActivationTypes.Sinusoid, ActivationTypes.RELU, ActivationTypes.HyperbolicTangens };

        private static NeatSettings MixedSettings() => new NeatSettings
        {
            HiddenActivations = Mixed.ToList(),
            OutputActivations = new[] { ActivationTypes.Identity, ActivationTypes.SoftSign }
        };

        private static IGenome Grown(NeatSettings settings, InnovationGenerator generator, int seed)
        {
            using var random = new SeededRandom(seed);
            IGenome genome = Genome.CreateMinimal(3, 2, generator, settings: settings);
            for (int i = 0; i < 30; i++)
            {
                genome.Mutate(i % 2 == 0 ? MutationType.Node : MutationType.Connection, generator, settings);
                genome.Mutate(MutationType.AllConnections, generator, settings);
            }

            return genome;
        }

        [Fact]
        public void CreateMinimal_GivesOutputsAnActivationAndLeavesInputsWithout()
        {
            Genome genome = Genome.CreateMinimal(3, 2, new InnovationGenerator());

            Assert.All(genome.Nodes.Values.Where(x => x.Type is NodeType.Input or NodeType.Bias), x => Assert.Null(x.Activation));
            Assert.All(genome.Nodes.Values.Where(x => x.Type == NodeType.Output), x => Assert.Equal(ActivationTypes.Sigmoid, x.Activation));

            using var seed = new SeededRandom(30);
            NeatSettings settings = MixedSettings();
            var activations = Enumerable.Range(0, 20)
                                        .SelectMany(_ => Genome.CreateMinimal(1, 5, new InnovationGenerator(), settings: settings).Nodes.Values)
                                        .Where(x => x.Type == NodeType.Output)
                                        .Select(x => x.Activation.Value)
                                        .Distinct()
                                        .OrderBy(x => x);
            Assert.Equal(settings.OutputActivations.OrderBy(x => x), activations);
        }

        [Fact]
        public void NodeMutation_PicksTheActivationFromTheHiddenActivations()
        {
            IGenome genome = Grown(MixedSettings(), new InnovationGenerator(), 31);

            var hidden = genome.Nodes.Values.Where(x => x.Type == NodeType.Hidden).ToList();
            Assert.True(hidden.Count >= 10);
            Assert.Equal(Mixed.OrderBy(x => x), hidden.Select(x => x.Activation.Value).Distinct().OrderBy(x => x));
        }

        [Fact]
        public void CloneAndCrossover_KeepTheActivations()
        {
            var generator = new InnovationGenerator();
            IGenome genome = Grown(MixedSettings(), generator, 32);

            IGenome clone = genome.Clone();
            Assert.Equal(genome.Nodes.Values.Select(x => (x.Innovation, x.Activation)), clone.Nodes.Values.Select(x => (x.Innovation, x.Activation)));

            IGenome other = genome.Clone();
            genome.Fitness = 2;
            other.Fitness = 1;
            using var seed = new SeededRandom(33);
            IGenome child = genome.Crossover(other, MixedSettings());
            Assert.Equal(genome.Nodes.Values.Select(x => (x.Innovation, x.Activation)), child.Nodes.Values.Select(x => (x.Innovation, x.Activation)));
        }

        [Fact]
        public void ToNetwork_UsesTheActivationOfEveryGene()
        {
            var settings = new NeatSettings { HiddenActivations = new[] { ActivationTypes.Gaussian }, OutputActivations = new[] { ActivationTypes.Identity } };
            var generator = new InnovationGenerator();
            Genome genome = Genome.CreateMinimal(1, 1, generator, withBias: false, settings: settings);
            genome.Connections.Values.Single().Weight = 1.5f;
            genome.Mutate(MutationType.Node, generator, settings);

            // input -(1)-> gaussian -(1.5)-> identity output
            Assert.Equal(1.5f * (float)Math.Exp(-0.7 * 0.7), (float)genome.ToNetwork().Propagate(0.7f)[0], 5);

            genome.Nodes.Values.Single(x => x.Type == NodeType.Hidden).Activation = ActivationTypes.Sinusoid;
            Assert.Equal(1.5f * (float)Math.Sin(0.7), (float)genome.ToNetwork().Propagate(0.7f)[0], 5);
        }

        [Fact]
        public void ToNetwork_UsesThePhenotypeSettingsForGenesWithoutActivation()
        {
            var generator = new InnovationGenerator();
            Genome genome = Genome.CreateMinimal(1, 1, generator, withBias: false);
            genome.Connections.Values.Single().Weight = 2f;
            genome.Nodes.Values.Single(x => x.Type == NodeType.Output).Activation = null;

            var phenotype = new PhenotypeSettings { OutputActivationType = ActivationTypes.HyperbolicTangens };

            Assert.Equal((float)Math.Tanh(2 * 0.3), (float)genome.ToNetwork(phenotype).Propagate(0.3f)[0], 5);
        }

        [Fact]
        public void Serialization_KeepsTheActivations()
        {
            IGenome genome = Grown(MixedSettings(), new InnovationGenerator(), 34);

            using var stream = new MemoryStream();
            GenomeSerializer.Save(genome, stream);
            stream.Position = 0;
            Genome loaded = GenomeSerializer.Load(stream);

            Assert.Equal(genome.Nodes.Values.Select(x => (x.Innovation, x.Type, x.Activation)), loaded.Nodes.Values.Select(x => (x.Innovation, x.Type, x.Activation)));
            Assert.Equal(genome.ToNetwork().Propagate(0.2f, -0.4f, 1.1f).Select(x => (float)x), loaded.ToNetwork().Propagate(0.2f, -0.4f, 1.1f).Select(x => (float)x));
        }

        [Fact]
        public void Serialization_LoadsVersionOneFilesWithoutActivations()
        {
            IGenome genome = Grown(MixedSettings(), new InnovationGenerator(), 35);
            using var stream = new MemoryStream();
            GenomeSerializer.Save(genome, stream);

            // a file as saved before activations were part of genes
            JsonNode json = JsonNode.Parse(stream.ToArray());
            json["version"] = 1;
            foreach (JsonNode node in json["genome"]["nodes"].AsArray())
            {
                node.AsObject().Remove("activation");
            }

            Genome loaded = GenomeSerializer.Load(new MemoryStream(Encoding.UTF8.GetBytes(json.ToJsonString())));

            Assert.All(loaded.Nodes.Values, x => Assert.Null(x.Activation));
            Assert.Equal(genome.Connections.Count, loaded.Connections.Count);

            json["genome"]["nodes"][0]["activation"] = "NotAnActivation";
            Assert.Throws<InvalidDataException>(() => GenomeSerializer.Load(new MemoryStream(Encoding.UTF8.GetBytes(json.ToJsonString()))));
        }

        private sealed class GrowthEvaluator : Evaluator
        {
            public GrowthEvaluator(IGenome genome, IInnovationGenerator generator, NeatSettings settings)
                : base(genome, generator, settings)
            { }

            public override float EvaluateGenome(IGenome genome) => genome.Nodes.Count + (genome.Connections.Values.Sum(x => Math.Abs(x.Weight)) / 100);
        }

        private static string Population(Evaluator evaluator)
        {
            return string.Join("|", evaluator.Genomes.Select(genome =>
            {
                using var stream = new MemoryStream();
                GenomeSerializer.Save(genome, stream);
                return Encoding.UTF8.GetString(stream.ToArray());
            }));
        }

        [Fact]
        public void Checkpoint_OfAMixedActivationRun_ContinuesIdentically()
        {
            GrowthEvaluator original;
            using (new SeededRandom(37))
            {
                var generator = new InnovationGenerator();
                NeatSettings settings = MixedSettings();
                settings.PopulationSize = 30;
                settings.AddNodeRate = 0.3f;
                original = new GrowthEvaluator(Genome.CreateMinimal(2, 1, generator, settings: settings), generator, settings);
                for (int i = 0; i < 10; i++) original.Evaluate();
            }

            using var stream = new MemoryStream();
            original.SaveCheckpoint(stream);
            stream.Position = 0;
            var fresh = new InnovationGenerator();
            var resumed = new GrowthEvaluator(Genome.CreateMinimal(2, 1, fresh), fresh, new NeatSettings());
            resumed.LoadCheckpoint(stream);

            Assert.Contains(original.Genomes.SelectMany(x => x.Nodes.Values), x => x.Activation is ActivationTypes.Gaussian or ActivationTypes.Sinusoid);
            Assert.Equal(Population(original), Population(resumed));

            using (new SeededRandom(38))
            {
                for (int i = 0; i < 5; i++) original.Evaluate();
            }

            using (new SeededRandom(38))
            {
                for (int i = 0; i < 5; i++) resumed.Evaluate();
            }

            Assert.Equal(Population(original), Population(resumed));
        }

        [Fact]
        public void MixedActivations_ExportToOnnxWithTheSameOutputs()
        {
            IGraphNetwork network = Grown(MixedSettings(), new InnovationGenerator(), 36).ToNetwork();
            float[][] samples = { new[] { 0f, 0f, 0f }, new[] { 1f, -1f, 0.5f }, new[] { -2.5f, 0.3f, 1.7f } };

            using var session = new InferenceSession(OnnxExporter.Export(network));
            using OrtValue input = OrtValue.CreateTensorValueFromMemory(samples.SelectMany(x => x).ToArray(), new long[] { samples.Length, 3 });
            using var runOptions = new RunOptions();
            using var results = session.Run(runOptions, new[] { "input" }, new[] { input }, session.OutputNames);
            float[] actual = results[0].GetTensorDataAsSpan<float>().ToArray();

            for (int i = 0; i < samples.Length; i++)
            {
                float[] expected = network.Propagate(samples[i][0], samples[i][1], samples[i][2]).Select(x => (float)x).ToArray();
                Assert.Equal(expected[0], actual[i * 2], 4);
                Assert.Equal(expected[1], actual[(i * 2) + 1], 4);
            }
        }
    }
}
