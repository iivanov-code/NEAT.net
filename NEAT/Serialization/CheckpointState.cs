using System.Collections.Generic;
using NEAT.Net.Abstrations.Interfaces;

namespace NEAT.Net.Serialization
{
    internal sealed record CheckpointState(uint Generation,
                                           float CompatibilityThreshold,
                                           InnovationGenerator InnovationGenerator,
                                           ICollection<IGenome> Genomes,
                                           ICollection<ISpecie> Species,
                                           IGenome FittestGenome);
}
