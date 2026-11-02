using System;
using NEAT.Net.Abstrations;

namespace NEAT.Net.Tests
{
    /// <summary>
    /// Makes a test reproducible by replacing the random source of the algorithm while it runs
    /// </summary>
    internal sealed class SeededRandom : IDisposable
    {
        private readonly Random previous;

        public SeededRandom(int seed)
        {
            previous = NEATUtils.Random;
            NEATUtils.Random = new Random(seed);
        }

        public void Dispose()
        {
            NEATUtils.Random = previous;
        }
    }

    [Xunit.CollectionDefinition(Name, DisableParallelization = true)]
    public sealed class RandomCollection
    {
        public const string Name = "NEAT random source";
    }
}
