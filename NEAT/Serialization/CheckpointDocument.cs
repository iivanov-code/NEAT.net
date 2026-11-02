using System.Collections.Generic;
using NEAT.Net.Abstrations;

namespace NEAT.Net.Serialization
{
    /// <summary>
    /// State of an evaluator. Genomes are stored once in <see cref="Genomes"/> and referred to by index, because the
    /// same genome can be part of the population, a specie and the best genome at the same time
    /// </summary>
    internal sealed class CheckpointDocument
    {
        public string Format { get; set; }

        public int Version { get; set; }

        public uint Generation { get; set; }

        /// <summary>
        /// Kept next to <see cref="Settings"/> for checkpoints that were saved before settings were part of them
        /// </summary>
        public float CompatibilityThreshold { get; set; }

        public NeatSettings Settings { get; set; }

        public InnovationDocument Innovations { get; set; }

        public List<GenomeDocument> Genomes { get; set; } = new List<GenomeDocument>();

        public List<int> Population { get; set; } = new List<int>();

        public List<SpecieDocument> Species { get; set; } = new List<SpecieDocument>();

        public int? FittestGenome { get; set; }
    }
}
