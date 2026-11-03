using System.Collections.Generic;
using System.Linq;
using NEAT.Net.Abstrations;
using NEAT.Net.Abstrations.Interfaces;

namespace NEAT.Net.Operators
{
    /// <summary>
    /// Matching connections before a random cut point come from the fitter parent, the ones from the cut point on from
    /// the weaker parent. The cut point is one of the matching innovation numbers; disjoint and excess genes come from
    /// the fitter parent as in <see cref="MultipointCrossover"/>
    /// </summary>
    public class SinglePointCrossover : MultipointCrossover
    {
        public override IGenome Cross(IGenome fitter, IGenome weaker, EvolutionContext context)
        {
            List<uint> matching = fitter.Connections.Keys.Where(weaker.Connections.ContainsKey).OrderBy(x => x).ToList();
            uint cut = matching.Count > 0 ? matching[context.Random.Next(matching.Count)] : 0;

            return Combine(fitter, weaker, context,
                           (child, fitterGene, weakerGene, _) => GenomeOperations.CopyConnection(child, fitterGene.Innovation < cut ? fitterGene : weakerGene));
        }
    }
}
