using NEAT.Net.Abstrations.Interfaces;
using System;

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
