using System.Collections.Generic;
using System.Linq;
using NEAT.Net.Abstrations;
using NEAT.Net.Abstrations.Enums;
using NEAT.Net.Abstrations.Interfaces;
using ActivationTypes = ANN.Net.Abstractions.Enums.ActivationTypes;

namespace NEAT.Net.Operators
{
    /// <summary>
    /// Gives one node another activation: a hidden node one of <see cref="NeatSettings.HiddenActivations"/>, an output
    /// node one of <see cref="NeatSettings.OutputActivations"/>. Nodes evolve only when their list holds more than one
    /// activation, so with single-entry lists the mutation has probability 0. Otherwise applied with
    /// <see cref="NeatSettings.ActivationMutationRate"/>
    /// </summary>
    public sealed class ActivationMutation : IMutation
    {
        public float GetProbability(EvolutionContext context)
        {
            NeatSettings settings = context.Settings;
            bool evolves = HasChoice(settings.HiddenActivations) || HasChoice(settings.OutputActivations);
            return evolves ? settings.ActivationMutationRate : 0f;
        }

        public void Apply(IGenome genome, EvolutionContext context)
        {
            NeatSettings settings = context.Settings;
            List<INodeGene> candidates = genome.Nodes.Values
                .Where(x => (x.Type == NodeType.Hidden && HasChoice(settings.HiddenActivations))
                         || (x.Type == NodeType.Output && HasChoice(settings.OutputActivations)))
                .ToList();

            if (candidates.Count == 0) return;

            INodeGene node = candidates.RandomElement(context.Random);
            IList<ActivationTypes> allowed = node.Type == NodeType.Hidden ? settings.HiddenActivations : settings.OutputActivations;
            List<ActivationTypes> options = allowed.Distinct().Where(x => x != node.Activation).ToList();

            if (options.Count > 0)
            {
                node.Activation = options.RandomElement(context.Random);
            }
        }

        private static bool HasChoice(IList<ActivationTypes> activations) => activations.Distinct().Skip(1).Any();
    }
}
