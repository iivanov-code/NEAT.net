using System;
using System.Collections.Generic;
using NEAT.Net.Abstrations.Enums;

namespace NEAT.Net.Abstrations.Interfaces
{
    public interface IGenome : IComparable<IGenome>, ICloneable<IGenome>
    {
        uint ID { get; }

        IDictionary<uint, IConnectionGene> Connections { get; }
        float Fitness { get; set; }

        event EventHandler<FitnessAssignedEventArgs> FitnessAssigned;

        IDictionary<uint, INodeGene> Nodes { get; }

        float CompatibilityDistance(IGenome other, float c1 = DefaultConstants.C1, float c2 = DefaultConstants.C2, float c3 = DefaultConstants.C3);

        /// <summary>
        /// Compatibility distance with the coefficients and the small genome size of <paramref name="settings"/>
        /// </summary>
        float CompatibilityDistance(IGenome other, NeatSettings settings);

        IGenome Crossover(IGenome other);

        IGenome Crossover(IGenome other, NeatSettings settings);

        void Mutate(MutationType mutationType, IInnovationGenerator generator);

        void Mutate(MutationType mutationType, IInnovationGenerator generator, NeatSettings settings);
    }
}
