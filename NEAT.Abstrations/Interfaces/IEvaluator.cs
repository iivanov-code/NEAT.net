﻿using System.Collections.Generic;

namespace NEAT.Net.Abstrations.Interfaces
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
