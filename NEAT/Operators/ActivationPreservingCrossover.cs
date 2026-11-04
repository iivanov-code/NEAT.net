using NEAT.Net.Abstrations;
using NEAT.Net.Abstrations.Interfaces;

namespace NEAT.Net.Operators
{
    /// <summary>
    /// Like <see cref="MultipointCrossover"/>, but every node keeps the activation of the fitter parent, for problems
    /// where mixing activations between parents does more harm than good
    /// </summary>
    public class ActivationPreservingCrossover : MultipointCrossover
    {
        protected override INodeGene InheritNode(INodeGene fitter, INodeGene weaker, EvolutionContext context)
        {
            return fitter.Clone();
        }
    }
}
