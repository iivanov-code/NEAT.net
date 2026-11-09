using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net;
using ANN.Net.Abstractions.Spiking;
using NEAT.Net.Abstrations.Interfaces;
using ActivationTypes = ANN.Net.Abstractions.Enums.ActivationTypes;
using NeatNodeType = NEAT.Net.Abstrations.Enums.NodeType;

namespace NEAT.Net.Networks
{
    /// <summary>
    /// Turns NEAT genomes into spiking ANN.Net networks, so that evolution can work on spiking networks directly
    /// </summary>
    public static class SpikingGenomeExtensions
    {
        /// <summary>
        /// Describes the genome as a spiking network. Inputs (ordered by node id) are layer 0; every hidden and output
        /// node becomes a spiking neuron, layered by depth (one more than its deepest input) and, within a depth, by
        /// activation. Enabled connections become synapses with the same weight; the connections of bias nodes become
        /// a constant input current of their targets. The outputs are the firing rates of the output nodes, ordered by
        /// node id
        /// </summary>
        public static SpikingGraphSettings ToSpikingSettings(this IGenome genome, SpikingPhenotypeSettings settings = null)
        {
            return SpikingLayout(genome, settings ?? new SpikingPhenotypeSettings()).Settings;
        }

        /// <summary>
        /// Builds the spiking network of the genome (see <see cref="ToSpikingSettings"/>). The network keeps the state
        /// of its neurons, so it must not be used by several threads at once
        /// </summary>
        public static ISpikingNetwork ToSpikingNetwork(this IGenome genome, SpikingPhenotypeSettings settings = null)
        {
            return NetworkFactory.BuildSpiking(genome.ToSpikingSettings(settings));
        }

        /// <summary>
        /// Writes the weights of a spiking network built from the genome with the same settings, for example after
        /// it learned with STDP, back into the genome's connections (Lamarckian evolution). Bias connections, which
        /// are no synapses, are left unchanged
        /// </summary>
        public static void ApplyWeights(this IGenome genome, ISpikingNetwork network, SpikingPhenotypeSettings settings = null)
        {
            if (genome == null) throw new ArgumentNullException(nameof(genome));
            if (network == null) throw new ArgumentNullException(nameof(network));

            var layout = SpikingLayout(genome, settings ?? new SpikingPhenotypeSettings());
            if (network.NeuronCount != layout.Settings.Layers.Sum(x => x.NeuronsCount))
                throw new ArgumentException("The network was not built from this genome", nameof(network));

            foreach (SpikingSynapse synapse in network.GetSynapses())
            {
                if (layout.Connections.TryGetValue((synapse.Source, synapse.Target), out IConnectionGene connection))
                {
                    connection.Weight = synapse.Weight;
                }
            }
        }

        private static (SpikingGraphSettings Settings, Dictionary<(int Source, int Target), IConnectionGene> Connections)
            SpikingLayout(IGenome genome, SpikingPhenotypeSettings settings)
        {
            if (genome == null) throw new ArgumentNullException(nameof(genome));

            List<INodeGene> nodes = genome.Nodes.Values.OrderBy(x => x.Innovation).ToList();
            List<IConnectionGene> connections = genome.Connections.Values.Where(x => x.Enabled).OrderBy(x => x.Innovation).ToList();

            var depth = new Dictionary<INodeGene, int>();
            int Depth(INodeGene node)
            {
                if (node.Type == NeatNodeType.Input || node.Type == NeatNodeType.Bias) return 0;
                if (depth.TryGetValue(node, out int known)) return known;

                // genomes are acyclic, so this ends
                int result = 1 + connections.Where(c => c.Output == node && c.Input.Type != NeatNodeType.Bias)
                                            .Select(c => Depth(c.Input))
                                            .DefaultIfEmpty(0)
                                            .Max();
                depth[node] = result;
                return result;
            }

            ActivationTypes Activation(INodeGene node) =>
                node.Activation ?? (node.Type == NeatNodeType.Output ? settings.OutputActivationType : settings.HiddenActivationType);

            List<INodeGene> inputs = nodes.Where(x => x.Type == NeatNodeType.Input).ToList();
            var layers = new List<SpikingGraphLayer> { new SpikingGraphLayer(inputs.Count, settings.InputActivation?.Invoke()) };
            var index = new Dictionary<INodeGene, int>();
            for (int i = 0; i < inputs.Count; i++) index[inputs[i]] = i;

            var groups = nodes.Where(x => x.Type == NeatNodeType.Hidden || x.Type == NeatNodeType.Output)
                              .GroupBy(x => (Depth: Depth(x), Activation: Activation(x)))
                              .OrderBy(x => x.Key.Depth).ThenBy(x => x.Key.Activation);
            foreach (var group in groups)
            {
                foreach (INodeGene node in group) index[node] = index.Count;
                layers.Add(new SpikingGraphLayer(group.Count(), settings.Activation(group.Key.Activation), settings.Learn));
            }

            var (spiking, byPair) = Wire(connections, index);
            spiking.Layers = layers;
            spiking.Outputs = nodes.Where(x => x.Type == NeatNodeType.Output).Select(x => index[x]).ToList();
            spiking.InputScaling = settings.InputScaling;
            spiking.TimeSteps = settings.TimeSteps;
            spiking.Plasticity = settings.Plasticity;
            return (spiking, byPair);
        }

        /// <summary>
        /// Synapses and bias currents of the enabled connections between the indexed neurons
        /// </summary>
        private static (SpikingGraphSettings Settings, Dictionary<(int Source, int Target), IConnectionGene> Connections)
            Wire(List<IConnectionGene> connections, Dictionary<INodeGene, int> index)
        {
            float[] bias = new float[index.Count];
            var synapses = new List<SpikingSynapse>();
            var byPair = new Dictionary<(int Source, int Target), IConnectionGene>();
            foreach (IConnectionGene connection in connections)
            {
                if (!index.TryGetValue(connection.Output, out int target)) continue;

                if (connection.Input.Type == NeatNodeType.Bias)
                {
                    bias[target] += connection.Weight;
                }
                else if (index.TryGetValue(connection.Input, out int source))
                {
                    synapses.Add(new SpikingSynapse(source, target, connection.Weight));
                    byPair[(source, target)] = connection;
                }
            }

            return (new SpikingGraphSettings { Synapses = synapses, Bias = bias }, byPair);
        }
    }
}
