using System.Linq;
using NEAT.Net.Abstrations;
using NEAT.Net.Abstrations.Interfaces;

namespace NEAT.Net.Operators
{
    /// <summary>
    /// Picks a random connection: a disabled one is enabled again, an enabled one is disabled if its target node keeps
    /// another enabled input. Enabling never closes a cycle, because new connections are checked against disabled
    /// connections too. Applied with <see cref="NeatSettings.ToggleEnableRate"/>
    /// </summary>
    public sealed class ToggleEnableMutation : IMutation
    {
        public float GetProbability(EvolutionContext context) => context.Settings.ToggleEnableRate;

        public void Apply(IGenome genome, EvolutionContext context)
        {
            if (genome.Connections.Count == 0) return;

            IConnectionGene connection = genome.Connections.Values.ToList().RandomElement(context.Random);

            if (!connection.Enabled)
            {
                connection.Enable();
            }
            else if (genome.Connections.Values.Any(x => x != connection && x.Enabled && x.Output.Innovation == connection.Output.Innovation))
            {
                connection.Disable();
            }
        }
    }
}
