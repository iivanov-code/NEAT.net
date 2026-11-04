using System;
using System.Collections.Generic;
using System.Linq;
using NEAT.Net.Abstrations;
using NEAT.Net.Abstrations.Interfaces;

namespace NEAT.Net.Operators
{
    /// <summary>
    /// Adds a connection between two unconnected nodes. Only feed-forward connections are created: the new
    /// connection may not close a cycle, counting disabled connections too since crossover can enable them again.
    /// Applied with <see cref="NeatSettings.AddConnectionRate"/>
    /// </summary>
    public sealed class AddConnectionMutation : IMutation
    {
        public float GetProbability(EvolutionContext context) => context.Settings.AddConnectionRate;

        public void Apply(IGenome genome, EvolutionContext context)
        {
            Random random = context.Random;
            List<INodeGene> sources = genome.Nodes.Values.Where(GenomeOperations.CanBeSource).ToList();
            List<INodeGene> targets = genome.Nodes.Values.Where(GenomeOperations.CanBeTarget).ToList();

            if (sources.Count == 0 || targets.Count == 0) return;

            for (int attempt = 0; attempt < context.Settings.AddConnectionAttempts; attempt++)
            {
                INodeGene fromNode = sources.RandomElement(random);
                INodeGene toNode = targets.RandomElement(random);

                if (fromNode.Innovation == toNode.Innovation) continue;

                bool hasConnection = genome.Connections.Values.Any(x => x.Input.Innovation == fromNode.Innovation && x.Output.Innovation == toNode.Innovation);
                if (hasConnection || GenomeOperations.IsReachable(genome, toNode.Innovation, fromNode.Innovation)) continue;

                uint innovation = context.InnovationGenerator.GetConnectionInnovationNumber(fromNode.Innovation, toNode.Innovation);
                genome.Connections.Add(innovation, new ConnectionGene(innovation, fromNode, toNode, GenomeOperations.RandomWeight(context.Settings, random)));
                return;
            }
        }
    }
}
