namespace NEAT.Net.Abstrations.Interfaces
{
    /// <summary>
    /// Changes a genome in place. The evaluator applies every mutation of its list to every offspring with the
    /// mutation's probability
    /// </summary>
    public interface IMutation
    {
        /// <summary>
        /// Chance that an offspring gets this mutation, usually a rate of <see cref="EvolutionContext.Settings"/>.
        /// A probability of 0 skips the mutation without drawing a random number
        /// </summary>
        float GetProbability(EvolutionContext context);

        void Apply(IGenome genome, EvolutionContext context);
    }
}
