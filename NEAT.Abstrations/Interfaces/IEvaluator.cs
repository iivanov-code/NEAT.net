using System.Collections.Generic;

namespace NEAT.Net.Abstrations.Interfaces
{
    public interface IEvaluator
    {
        /// <summary>
        /// Snapshot of the best genome found in any generation so far
        /// </summary>
        IGenome FittestGenome { get; }

        /// <summary>
        /// Parameters of the run; changes take effect from the next generation
        /// </summary>
        NeatSettings Settings { get; }

        ushort PopulationSize { get; }

        uint Generation { get; }

        ICollection<ISpecie> Species { get; }

        ICollection<IGenome> Genomes { get; }

        float EvaluateGenome(IGenome genome);

        /// <summary>
        /// Runs one generation: evaluation, speciation, elitism and breeding
        /// </summary>
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
