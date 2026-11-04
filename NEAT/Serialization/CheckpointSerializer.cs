using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using NEAT.Net.Abstrations.Interfaces;

namespace NEAT.Net.Serialization
{
    /// <summary>
    /// Saves and loads the state of an <see cref="Evaluator"/>
    /// </summary>
    internal static class CheckpointSerializer
    {
        private const string FormatName = "NEAT.Net checkpoint";
        /// <summary>
        /// Version 2 holds genomes with node activations (genome format 2); version 1 checkpoints still load
        /// </summary>
        private const int FormatVersion = 2;

        public static void Save(Stream stream, CheckpointState state)
        {
            ArgumentNullException.ThrowIfNull(stream);

            var genomes = new List<GenomeDocument>();
            var indexes = new Dictionary<IGenome, int>(ReferenceEqualityComparer.Instance);

            int Index(IGenome genome)
            {
                if (!indexes.TryGetValue(genome, out int index))
                {
                    index = genomes.Count;
                    indexes.Add(genome, index);
                    genomes.Add(GenomeSerializer.ToDocument(genome));
                }

                return index;
            }

            var document = new CheckpointDocument
            {
                Format = FormatName,
                Version = FormatVersion,
                Generation = state.Generation,
                CompatibilityThreshold = state.Settings.CompatibilityThreshold,
                Settings = state.Settings,
                Innovations = state.InnovationGenerator.ToDocument(),
                Population = state.Genomes.Select(Index).ToList(),
                Species = state.Species.Select(x => new SpecieDocument
                {
                    Mascot = Index(x.Mascot),
                    Members = x.Genomes.Select(Index).ToList(),
                    BestFitness = x.BestFitness,
                    Stagnation = x.Stagnation
                }).ToList(),
                FittestGenome = state.FittestGenome == null ? null : Index(state.FittestGenome),
                Genomes = genomes
            };

            JsonSerializer.Serialize(stream, document, GenomeSerializer.JsonOptions);
        }

        public static CheckpointState Load(Stream stream)
        {
            CheckpointDocument document = GenomeSerializer.Deserialize<CheckpointDocument>(stream);

            if (document == null || !string.Equals(document.Format, FormatName, StringComparison.Ordinal))
                throw new InvalidDataException("The file is not a NEAT.Net checkpoint");
            if (document.Version < 1 || document.Version > FormatVersion)
                throw new InvalidDataException($"Unsupported checkpoint file version {document.Version}");
            if (document.Innovations == null)
                throw new InvalidDataException("The checkpoint has no innovation history");

            Genome[] genomes = (document.Genomes ?? new List<GenomeDocument>()).Select(GenomeSerializer.FromDocument).ToArray();

            Genome At(int index)
            {
                if (index < 0 || index >= genomes.Length)
                    throw new InvalidDataException($"Unknown genome index {index}");

                return genomes[index];
            }

            List<IGenome> population = (document.Population ?? new List<int>()).Select(x => (IGenome)At(x)).ToList();
            List<ISpecie> species = (document.Species ?? new List<SpecieDocument>())
                .Select(x => (ISpecie)new Specie(At((x ?? throw new InvalidDataException("A specie is missing")).Mascot),
                                                 (x.Members ?? new List<int>()).Select(At),
                                                 x.BestFitness,
                                                 x.Stagnation))
                .ToList();

            try
            {
                document.Settings?.Validate();
            }
            catch (ArgumentException e)
            {
                throw new InvalidDataException("The checkpoint holds invalid settings", e);
            }

            return new CheckpointState(document.Generation,
                                       document.Settings,
                                       document.CompatibilityThreshold,
                                       InnovationGenerator.FromDocument(document.Innovations),
                                       population,
                                       species,
                                       document.FittestGenome is int fittest ? At(fittest) : null);
        }
    }
}
