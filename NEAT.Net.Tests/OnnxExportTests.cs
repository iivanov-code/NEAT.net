using System.Linq;
using ANN.Net;
using ANN.Net.Abstractions.Interfaces;
using Microsoft.ML.OnnxRuntime;
using NEAT.Net.Abstrations.Enums;
using NEAT.Net.Abstrations.Interfaces;
using NEAT.Net.Networks;
using Xunit;

namespace NEAT.Net.Tests
{
    [Collection(RandomCollection.Name)]
    public class OnnxExportTests
    {
        [Fact]
        public void EvolvedNetwork_RunsInOnnxRuntimeWithTheSameOutputs()
        {
            using var seed = new SeededRandom(11);
            var generator = new InnovationGenerator();
            IGenome genome = Genome.CreateMinimal(3, 2, generator);
            for (int i = 0; i < 40; i++)
            {
                genome.Mutate(i % 3 == 0 ? MutationType.Node : MutationType.Connection, generator);
                genome.Mutate(MutationType.AllConnections, generator);
            }

            IGraphNetwork network = genome.ToNetwork();
            float[][] samples = { new[] { 0f, 0f, 0f }, new[] { 1f, -1f, 0.5f }, new[] { -2.5f, 0.3f, 1.7f } };

            using var session = new InferenceSession(OnnxExporter.Export(network));
            using OrtValue input = OrtValue.CreateTensorValueFromMemory(samples.SelectMany(x => x).ToArray(), new long[] { samples.Length, 3 });
            using var runOptions = new RunOptions();
            using var results = session.Run(runOptions, new[] { "input" }, new[] { input }, session.OutputNames);
            float[] actual = results[0].GetTensorDataAsSpan<float>().ToArray();

            for (int i = 0; i < samples.Length; i++)
            {
                float[] expected = network.Propagate(samples[i][0], samples[i][1], samples[i][2]).Select(x => (float)x).ToArray();
                Assert.Equal(expected[0], actual[(i * 2) + 0], 5);
                Assert.Equal(expected[1], actual[(i * 2) + 1], 5);
            }
        }
    }
}
