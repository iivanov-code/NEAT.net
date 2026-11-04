using System.Collections.Generic;
using System.Linq;
using NEAT.Net.Abstrations;
using NEAT.Net.Abstrations.Enums;
using NEAT.Net.Abstrations.Interfaces;

namespace NEAT.Net.Operators
{
    /// <summary>
    /// Removes a random hidden node together with every connection gene from or to it.
    /// Applied with <see cref="NeatSettings.RemoveNodeRate"/>
    /// </summary>
    public sealed class RemoveNodeMutation : IMutation
    {
        public float GetProbability(EvolutionContext context) => context.Settings.RemoveNodeRate;

        public void Apply(IGenome genome, EvolutionContext context)
        {
            List<INodeGene> hidden = genome.Nodes.Values.Where(x => x.Type == NodeType.Hidden).ToList();
            if (hidden.Count == 0) return;

            INodeGene node = hidden.RandomElement(context.Random);
            List<uint> connections = genome.Connections.Values
                .Where(x => x.Input.Innovation == node.Innovation || x.Output.Innovation == node.Innovation)
                .Select(x => x.Innovation)
                .ToList();

            foreach (uint innovation in connections)
            {
                genome.Connections.Remove(innovation);
            }

            genome.Nodes.Remove(node.Innovation);
        }
    }
}
