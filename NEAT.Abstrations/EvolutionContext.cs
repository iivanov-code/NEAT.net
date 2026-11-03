using System;
using NEAT.Net.Abstrations.Interfaces;

namespace NEAT.Net.Abstrations
{
    /// <summary>
    /// What mutation and crossover operators work with: the innovation history, the parameters of the run and the
    /// random source
    /// </summary>
    public sealed class EvolutionContext
    {
        public EvolutionContext(IInnovationGenerator innovationGenerator, NeatSettings settings, Random random)
        {
            InnovationGenerator = innovationGenerator;
            Settings = settings ?? throw new ArgumentNullException(nameof(settings));
            Random = random ?? throw new ArgumentNullException(nameof(random));
        }

        /// <summary>
        /// Source of historical markings for structural mutations; may be null in a context used only for crossover
        /// </summary>
        public IInnovationGenerator InnovationGenerator { get; }

        public NeatSettings Settings { get; }

        public Random Random { get; }
    }
}
