using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using NEAT.Net.Abstrations.Enums;
using NEAT.Net.Abstrations.Interfaces;
using NEAT.Net.Serialization;

namespace NEAT.Net
{
    /// <summary>
    /// Saves genomes to JSON and loads them back: node genes, connection genes (innovation, endpoints, weight,
    /// enabled) and fitness. To continue a run, save a checkpoint with <see cref="Evaluator.SaveCheckpoint(string)"/>,
    /// which also keeps the innovation history; new mutations of a loaded genome must use a generator that knows
    /// its innovation numbers
    /// </summary>
    public static class GenomeSerializer
    {
        private const string FormatName = "NEAT.Net genome";
        /// <summary>
        /// Version 2 added the activation of node genes; version 1 files still load
        /// </summary>
        internal const int FormatVersion = 2;

        internal static readonly JsonSerializerOptions JsonOptions = new JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
            NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals,
            Converters = { new JsonStringEnumConverter() }
        };

        public static void Save(IGenome genome, string filePath)
        {
            ArgumentException.ThrowIfNullOrEmpty(filePath);

            using FileStream stream = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.None);
            Save(genome, stream);
        }

        public static void Save(IGenome genome, Stream stream)
        {
            ArgumentNullException.ThrowIfNull(genome);
            ArgumentNullException.ThrowIfNull(stream);

            JsonSerializer.Serialize(stream, new GenomeFileDocument { Format = FormatName, Version = FormatVersion, Genome = ToDocument(genome) }, JsonOptions);
        }

        public static Genome Load(string filePath)
        {
            ArgumentException.ThrowIfNullOrEmpty(filePath);

            using FileStream stream = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read);
            return Load(stream);
        }

        public static Genome Load(Stream stream)
        {
            GenomeFileDocument document = Deserialize<GenomeFileDocument>(stream);

            if (document == null || !string.Equals(document.Format, FormatName, StringComparison.Ordinal))
                throw new InvalidDataException("The file is not a NEAT.Net genome");
            if (document.Version < 1 || document.Version > FormatVersion)
                throw new InvalidDataException($"Unsupported genome file version {document.Version}");

            return FromDocument(document.Genome);
        }

        internal static T Deserialize<T>(Stream stream)
        {
            ArgumentNullException.ThrowIfNull(stream);

            try
            {
                return JsonSerializer.Deserialize<T>(stream, JsonOptions);
            }
            catch (JsonException e)
            {
                throw new InvalidDataException("The file is not a valid NEAT.Net file", e);
            }
        }

        internal static GenomeDocument ToDocument(IGenome genome)
        {
            return new GenomeDocument
            {
                Fitness = genome.Fitness,
                // genes keep the order of the genome: mutations pick genes by position, so a loaded genome
                // evolves exactly like the original for the same random numbers
                Nodes = genome.Nodes.Values
                              .Select(x => new NodeGeneDocument { Id = x.Innovation, Type = x.Type, Activation = x.Activation })
                              .ToList(),
                Connections = genome.Connections.Values
                                    .Select(x => new ConnectionGeneDocument
                                    {
                                        Innovation = x.Innovation,
                                        From = x.Input.Innovation,
                                        To = x.Output.Innovation,
                                        Weight = x.Weight,
                                        Enabled = x.Enabled
                                    })
                                    .ToList()
            };
        }

        internal static Genome FromDocument(GenomeDocument document)
        {
            if (document == null) throw new InvalidDataException("A genome is missing");

            var genome = new Genome();

            foreach (NodeGeneDocument node in document.Nodes ?? new List<NodeGeneDocument>())
            {
                if (node == null || !Enum.IsDefined(node.Type) || (node.Activation.HasValue && !Enum.IsDefined(node.Activation.Value)))
                    throw new InvalidDataException("Invalid node gene");
                if (!genome.Nodes.TryAdd(node.Id, new NodeGene(node.Id, node.Type, node.Activation)))
                    throw new InvalidDataException($"Duplicate node gene {node.Id}");
            }

            foreach (ConnectionGeneDocument connection in document.Connections ?? new List<ConnectionGeneDocument>())
            {
                if (connection == null) throw new InvalidDataException("Invalid connection gene");

                if (!genome.Nodes.TryGetValue(connection.From, out INodeGene from) || !genome.Nodes.TryGetValue(connection.To, out INodeGene to))
                    throw new InvalidDataException($"Connection gene {connection.Innovation} refers to an unknown node");
                if (from.Type == NodeType.Output || to.Type == NodeType.Input || to.Type == NodeType.Bias)
                    throw new InvalidDataException($"Connection gene {connection.Innovation} has an invalid direction");

                var gene = new ConnectionGene(connection.Innovation, from, to, connection.Weight, connection.Enabled);
                if (!genome.Connections.TryAdd(connection.Innovation, gene))
                    throw new InvalidDataException($"Duplicate connection gene {connection.Innovation}");
            }

            genome.Fitness = document.Fitness;
            return genome;
        }
    }
}
