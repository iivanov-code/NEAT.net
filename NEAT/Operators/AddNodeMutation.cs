using System.Collections.Generic;
using System.Linq;
using NEAT.Net.Abstrations;
using NEAT.Net.Abstrations.Enums;
using NEAT.Net.Abstrations.Interfaces;

namespace NEAT.Net.Operators
{
    /// <summary>
    /// Splits an enabled connection: the old connection is disabled, the connection into the new node gets
    /// weight 1 and the connection out of it the old weight, so the network behaves almost as before.
    /// The new node gets a random activation of <see cref="NeatSettings.HiddenActivations"/>.
    /// Applied with <see cref="NeatSettings.AddNodeRate"/>
    /// </summary>
    public sealed class AddNodeMutation : IMutation
    {
        public float GetProbability(EvolutionContext context) => context.Settings.AddNodeRate;

        public void Apply(IGenome genome, EvolutionContext context)
        {
            List<IConnectionGene> enabled = genome.Connections.Values.Where(x => x.Enabled).ToList();
            if (enabled.Count == 0) return;

            IConnectionGene connection = enabled.RandomElement(context.Random);
            IInnovationGenerator generator = context.InnovationGenerator;

            uint nodeId = generator.GetNodeInnovationNumber(connection.Innovation);
            if (genome.Nodes.ContainsKey(nodeId))
            {
                // this genome split the same connection before (it was enabled again by crossover)
                nodeId = generator.NewNodeInnovationNumber();
            }

            connection.Disable();
            INodeGene newNode = new NodeGene(nodeId, NodeType.Hidden, GenomeOperations.PickActivation(context.Settings.HiddenActivations, context.Random));
            genome.Nodes.Add(nodeId, newNode);

            uint innovation = generator.GetConnectionInnovationNumber(connection.Input.Innovation, newNode.Innovation);
            genome.Connections.Add(innovation, new ConnectionGene(innovation, connection.Input, newNode, 1f));

            innovation = generator.GetConnectionInnovationNumber(newNode.Innovation, connection.Output.Innovation);
            genome.Connections.Add(innovation, new ConnectionGene(innovation, newNode, connection.Output, connection.Weight));
        }
    }
}
