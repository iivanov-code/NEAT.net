using System;
using System.Linq;
using ANN.Net;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Settings;
using NEAT.Net.Abstrations.Interfaces;
using AnnNodeType = ANN.Net.Abstractions.Enums.NodeType;
using NeatNodeType = NEAT.Net.Abstrations.Enums.NodeType;

namespace NEAT.Net.Networks
{
    /// <summary>
    /// Turns NEAT genomes into ANN.Net networks (the phenotype)
    /// </summary>
    public static class GenomeNetworkExtensions
    {
        /// <summary>
        /// Describes the genome as a graph: node genes become neurons with the same id and enabled connection genes
        /// become synapses with the innovation number as id. Inputs and outputs are ordered by node id, which for
        /// <see cref="Genome.CreateMinimal"/> is the order they were created in. Every node gets the activation of
        /// its gene; nodes without one get the activation of <paramref name="settings"/> for their type
        /// </summary>
        public static GraphNetworkSettings ToGraphSettings(this IGenome genome, PhenotypeSettings settings = null)
        {
            if (genome == null) throw new ArgumentNullException(nameof(genome));
            settings ??= new PhenotypeSettings();

            return new GraphNetworkSettings
            {
                Nodes = genome.Nodes.Values
                              .OrderBy(x => x.Innovation)
                              .Select(x => new GraphNode(x.Innovation, ToAnnNodeType(x.Type), x.Activation))
                              .ToList(),
                Connections = genome.Connections.Values
                                    .OrderBy(x => x.Innovation)
                                    .Select(x => new GraphConnection(x.Innovation, x.Input.Innovation, x.Output.Innovation, x.Weight, x.Enabled))
                                    .ToList(),
                InputActivationType = settings.InputActivationType,
                HiddenActivationType = settings.HiddenActivationType,
                OutputActivationType = settings.OutputActivationType,
                OptimizerType = settings.OptimizerType,
                LossType = settings.LossType,
                OutputApplySoftMax = settings.OutputApplySoftMax,
                IdentityGenerator = settings.IdentityGenerator
            };
        }

        /// <summary>
        /// Builds the ANN.Net network of the genome
        /// </summary>
        public static IGraphNetwork ToNetwork(this IGenome genome, PhenotypeSettings settings = null)
        {
            return NetworkFactory.BuildGraph(genome.ToGraphSettings(settings));
        }

        /// <summary>
        /// Inference-only network of the genome: computes what <see cref="ToNetwork"/> computes, but is compiled
        /// straight into a few weight matrices without building neurons and synapses, so it is several times faster to
        /// build and to evaluate and uses a fraction of the memory. It cannot be trained; use <see cref="ToNetwork"/> for
        /// backpropagation. It keeps no state between calls, so it can be evaluated from several threads at once
        /// </summary>
        public static ICompactNetwork ToCompactNetwork(this IGenome genome, PhenotypeSettings settings = null)
        {
            return NetworkCompactor.CompactGraph(genome.ToGraphSettings(settings));
        }

        /// <summary>
        /// Writes the weights of a network, for example after training it with backpropagation, back into the
        /// genome (Lamarckian evolution). Connections the network does not contain are left unchanged
        /// </summary>
        public static void ApplyWeights(this IGenome genome, IGraphNetwork network)
        {
            if (genome == null) throw new ArgumentNullException(nameof(genome));
            if (network == null) throw new ArgumentNullException(nameof(network));

            foreach (var weight in network.ConnectionWeights)
            {
                if (genome.Connections.TryGetValue(weight.Key, out IConnectionGene connection))
                {
                    connection.Weight = weight.Value;
                }
            }
        }

        private static AnnNodeType ToAnnNodeType(NeatNodeType type)
        {
            switch (type)
            {
                case NeatNodeType.Input:
                    return AnnNodeType.INPUT;
                case NeatNodeType.Hidden:
                    return AnnNodeType.HIDDEN;
                case NeatNodeType.Output:
                    return AnnNodeType.OUTPUT;
                case NeatNodeType.Bias:
                    return AnnNodeType.BIAS;
                default:
                    throw new ArgumentException($"Unknown node type {type}", nameof(type));
            }
        }
    }
}
