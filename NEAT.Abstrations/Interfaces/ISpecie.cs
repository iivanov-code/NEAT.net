using System;
using System.Collections.Generic;

namespace NEAT.Net.Abstrations.Interfaces
{
    public interface ISpecie : IReadOnlyList<IGenome>
    {
        /// <summary>
        /// Representative every new genome is compared with
        /// </summary>
        IGenome Mascot { get; }

        IGenome FittestGenome { get; }

        IList<IGenome> Genomes { get; }

        float AverageFitness { get; }

        /// <summary>
        /// Best fitness the specie has ever reached
        /// </summary>
        float BestFitness { get; }

        /// <summary>
        /// Generations since <see cref="BestFitness"/> last improved
        /// </summary>
        int Stagnation { get; }

        bool TryAddGenome(IGenome genome, float dt = DefaultConstants.DT, float c1 = DefaultConstants.C1, float c2 = DefaultConstants.C2, float c3 = DefaultConstants.C3);

        /// <summary>
        /// Updates <see cref="BestFitness"/> and <see cref="Stagnation"/> once the members have been evaluated
        /// </summary>
        void UpdateStagnation();

        /// <summary>
        /// Picks a random member as the representative for the next generation and removes all members
        /// </summary>
        void StartNewGeneration(Random random);
    }
}
