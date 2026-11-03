using System.Linq;
using NEAT.Net.Abstrations;
using NEAT.Net.Abstrations.Interfaces;

namespace NEAT.Net.Operators
{
    /// <summary>
    /// Removes a random connection gene, keeping genomes small. Nodes left without connections stay in the genome and
    /// are left out of the network. Applied with <see cref="NeatSettings.RemoveConnectionRate"/>
    /// </summary>
    public sealed class RemoveConnectionMutation : IMutation
    {
        public float GetProbability(EvolutionContext context) => context.Settings.RemoveConnectionRate;

        public void Apply(IGenome genome, EvolutionContext context)
        {
            if (genome.Connections.Count == 0) return;

            IConnectionGene connection = genome.Connections.Values.ToList().RandomElement(context.Random);
            genome.Connections.Remove(connection.Innovation);
        }
    }
}
