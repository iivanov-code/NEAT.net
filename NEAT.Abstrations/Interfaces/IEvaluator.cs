using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NEAT.Abstrations.Interfaces
{
    public interface IEvaluator
    {
        IGenome FittestGenome { get; }
        ushort PopulationSize { get; }
        ICollection<ISpecie> Species { get; }

        ICollection<IGenome> Genomes { get; }

        abstract float EvaluateGenome(IGenome genome);

        void Evaluate();

        void EvaluateGenomes();

        void CategorizeIntoSpecies();

        /// <summary>
        /// Puts best genomes into next generation
        /// </summary>
        void PutGenomesIntoNextGeneration();

        void BreedGenomes();
    }
}
