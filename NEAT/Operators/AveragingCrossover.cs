using NEAT.Net.Abstrations;
using NEAT.Net.Abstrations.Interfaces;

namespace NEAT.Net.Operators
{
    /// <summary>
    /// Like <see cref="MultipointCrossover"/>, but a matching connection gets the average weight of both parents
    /// ("mate multipoint avg" of the original NEAT implementation)
    /// </summary>
    public class AveragingCrossover : MultipointCrossover
    {
        protected override IConnectionGene InheritMatching(IGenome child, IConnectionGene fitter, IConnectionGene weaker, EvolutionContext context)
        {
            IConnectionGene connection = GenomeOperations.CopyConnection(child, fitter);
            connection.Weight = (fitter.Weight + weaker.Weight) / 2;
            return connection;
        }
    }
}
