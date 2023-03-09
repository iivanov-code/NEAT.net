using System;
using NEAT.Net.Abstrations.Interfaces;

namespace NEAT.Net.Abstrations
{
    public class FitnessAssignedEventArgs : EventArgs
    {
        public readonly IGenome Genome;
        public FitnessAssignedEventArgs(IGenome genome)
        {
            Genome = genome;
        }
    }
}
