using System.Collections.Generic;
using NEAT.Net.Abstrations;
using NEAT.Net.Abstrations.Interfaces;

namespace NEAT.Net.Serialization
{
    /// <param name="Settings">Null for checkpoints saved before settings were part of them</param>
    /// <param name="CompatibilityThreshold">The only setting older checkpoints hold</param>
    internal sealed record CheckpointState(uint Generation,
                                           NeatSettings Settings,
                                           float CompatibilityThreshold,
                                           InnovationGenerator InnovationGenerator,
                                           ICollection<IGenome> Genomes,
                                           ICollection<ISpecie> Species,
                                           IGenome FittestGenome);
}
