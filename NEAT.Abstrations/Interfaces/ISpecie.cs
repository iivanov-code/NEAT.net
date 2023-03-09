using System.Collections.Generic;

namespace NEAT.Net.Abstrations.Interfaces
{
    public interface ISpecie : IReadOnlyList<IGenome>
    {
        IGenome Mascot { get; }
        IGenome FittestGenome { get; }
        IList<IGenome> Genomes { get; }
        float AverageFitness { get; }
        bool TryAddGenome(IGenome genome, float dt = DefaultConstants.DT, float c1 = DefaultConstants.C1, float c2 = DefaultConstants.C2, float c3 = DefaultConstants.C3);
    }
}
