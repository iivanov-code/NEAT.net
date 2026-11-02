namespace NEAT.Net.Tests
{
    /// <summary>
    /// Tests that replace <see cref="NEAT.Net.Abstrations.NEATUtils.Random"/> must not run in parallel
    /// </summary>
    [Xunit.CollectionDefinition(Name, DisableParallelization = true)]
    public sealed class RandomCollection
    {
        public const string Name = "NEAT random source";

        private RandomCollection()
        { }
    }
}
