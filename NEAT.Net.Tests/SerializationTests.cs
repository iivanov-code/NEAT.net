using System;
using System.IO;
using System.Linq;
using System.Text;
using NEAT.Net.Abstrations;
using NEAT.Net.Abstrations.Enums;
using NEAT.Net.Abstrations.Interfaces;
using NEAT.Net.Networks;
using Xunit;

namespace NEAT.Net.Tests
{
    [Collection(RandomCollection.Name)]
    public class SerializationTests
    {
        private sealed class SizeEvaluator : Evaluator
        {
            public SizeEvaluator(IGenome genome, IInnovationGenerator generator)
                : base(genome, generator, 40)
            { }

            // deterministic, and it rewards growth so the population keeps changing
            public override float EvaluateGenome(IGenome genome)
            {
                return genome.Connections.Values.Count(x => x.Enabled) + (genome.Connections.Values.Sum(x => Math.Abs(x.Weight)) / 100);
            }
        }

        private static IGenome Mutated(InnovationGenerator generator, int seed)
        {
            using var random = new SeededRandom(seed);
            IGenome genome = Genome.CreateMinimal(3, 2, generator);
            for (int i = 0; i < 30; i++)
            {
                genome.Mutate(i % 3 == 0 ? MutationType.Node : MutationType.Connection, generator);
                genome.Mutate(MutationType.AllConnections, generator);
            }

            genome.Fitness = 12.5f;
            return genome;
        }

        private static string Json(IGenome genome)
        {
            using var stream = new MemoryStream();
            GenomeSerializer.Save(genome, stream);
            return Encoding.UTF8.GetString(stream.ToArray());
        }

        private static Genome RoundTrip(IGenome genome)
        {
            using var stream = new MemoryStream();
            GenomeSerializer.Save(genome, stream);
            stream.Position = 0;
            return GenomeSerializer.Load(stream);
        }

        private static string Snapshot(Evaluator evaluator)
        {
            return string.Join("|", evaluator.Genomes.Select(Json))
                + $"#{evaluator.Generation}#{evaluator.Species.Count}#{evaluator.FittestGenome?.Fitness}"
                + string.Join(",", evaluator.Species.Select(x => $"{x.Count}/{x.Stagnation}/{x.BestFitness}"));
        }

        [Fact]
        public void Genome_RoundTrip_KeepsEveryGene()
        {
            var generator = new InnovationGenerator();
            IGenome genome = Mutated(generator, 1);
            Assert.Contains(genome.Connections.Values, x => !x.Enabled);

            Genome loaded = RoundTrip(genome);

            Assert.Equal(12.5f, loaded.Fitness);
            Assert.Equal(genome.Nodes.Values.Select(x => (x.Innovation, x.Type)), loaded.Nodes.Values.Select(x => (x.Innovation, x.Type)));
            Assert.Equal(genome.Connections.Values.Select(x => (x.Innovation, x.Input.Innovation, x.Output.Innovation, x.Weight, x.Enabled)),
                         loaded.Connections.Values.Select(x => (x.Innovation, x.Input.Innovation, x.Output.Innovation, x.Weight, x.Enabled)));
            foreach (IConnectionGene connection in loaded.Connections.Values)
            {
                Assert.Same(loaded.Nodes[connection.Input.Innovation], connection.Input);
                Assert.Same(loaded.Nodes[connection.Output.Innovation], connection.Output);
            }

            Assert.Equal(genome.ToNetwork().Propagate(0.5f, -1f, 2f).Select(x => (float)x), loaded.ToNetwork().Propagate(0.5f, -1f, 2f).Select(x => (float)x));
            Assert.Equal(0f, genome.CompatibilityDistance(loaded));
        }

        [Fact]
        public void Genome_SavedToFile_KeepsEvolvingWithTheSameGenerator()
        {
            var generator = new InnovationGenerator();
            IGenome genome = Mutated(generator, 2);
            string file = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".json");

            try
            {
                GenomeSerializer.Save(genome, file);
                Genome loaded = GenomeSerializer.Load(file);

                using var seed = new SeededRandom(3);
                for (int i = 0; i < 20; i++)
                {
                    loaded.Mutate(i % 2 == 0 ? MutationType.Node : MutationType.Connection, generator);
                }

                Assert.True(loaded.Nodes.Count > genome.Nodes.Count);
                Assert.NotEmpty(loaded.Crossover(genome).Connections);
            }
            finally
            {
                File.Delete(file);
            }
        }

        [Fact]
        public void Genome_Load_RejectsInvalidFiles()
        {
            static Genome LoadText(string text) => GenomeSerializer.Load(new MemoryStream(Encoding.UTF8.GetBytes(text)));

            string json = Json(Mutated(new InnovationGenerator(), 4));

            Assert.Throws<InvalidDataException>(() => LoadText("not json"));
            Assert.Throws<InvalidDataException>(() => LoadText("{ \"format\": \"something else\", \"version\": 1 }"));
            Assert.Throws<InvalidDataException>(() => LoadText(json.Replace("\"version\": 2", "\"version\": 99", StringComparison.Ordinal)));
            Assert.Throws<InvalidDataException>(() => LoadText(json.Replace("\"from\": ", "\"from\": 9999", StringComparison.Ordinal)));
        }

        [Fact]
        public void Checkpoint_ContinuesTheRunExactlyWhereItStopped()
        {
            Evaluator original;
            using (new SeededRandom(5))
            {
                var generator = new InnovationGenerator();
                original = new SizeEvaluator(Genome.CreateMinimal(3, 2, generator), generator);
                for (int i = 0; i < 15; i++)
                {
                    original.Evaluate();
                }
            }

            using var stream = new MemoryStream();
            original.SaveCheckpoint(stream);
            stream.Position = 0;

            // a fresh evaluator with its own generator takes over the saved run
            var fresh = new InnovationGenerator();
            Evaluator resumed = new SizeEvaluator(Genome.CreateMinimal(3, 2, fresh), fresh);
            resumed.LoadCheckpoint(stream);

            Assert.Equal(Snapshot(original), Snapshot(resumed));

            // with the same random numbers both runs go on identically, so no state was lost
            using (new SeededRandom(6))
            {
                for (int i = 0; i < 10; i++) original.Evaluate();
            }

            using (new SeededRandom(6))
            {
                for (int i = 0; i < 10; i++) resumed.Evaluate();
            }

            Assert.Equal(25u, resumed.Generation);
            Assert.Equal(Snapshot(original), Snapshot(resumed));
        }

        [Fact]
        public void Checkpoint_SavedToFile_LoadsBack()
        {
            var generator = new InnovationGenerator();
            var evaluator = new SizeEvaluator(Genome.CreateMinimal(2, 1, generator), generator);
            evaluator.Evaluate();
            string file = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".json");

            try
            {
                evaluator.SaveCheckpoint(file);
                var resumed = new SizeEvaluator(Genome.CreateMinimal(2, 1, new InnovationGenerator()), new InnovationGenerator());
                resumed.LoadCheckpoint(file);

                Assert.Equal(Snapshot(evaluator), Snapshot(resumed));
                Assert.Throws<InvalidDataException>(() => GenomeSerializer.Load(file));
            }
            finally
            {
                File.Delete(file);
            }
        }
    }
}
