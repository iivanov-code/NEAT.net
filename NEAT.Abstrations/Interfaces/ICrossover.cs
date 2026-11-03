namespace NEAT.Net.Abstrations.Interfaces
{
    /// <summary>
    /// Combines two parents into a new genome. The evaluator passes the fitter parent first (for equally fit
    /// parents it picks one at random); the parents must not be changed
    /// </summary>
    public interface ICrossover
    {
        IGenome Cross(IGenome fitter, IGenome weaker, EvolutionContext context);
    }
}
