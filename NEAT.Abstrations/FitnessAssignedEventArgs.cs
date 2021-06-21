using System;

namespace NEAT.Abstrations
{
    public class FitnessAssignedEventArgs : EventArgs
    {
        public readonly IGenome Genome;
        public FitnessAssignedEventArgs(IGenome genome)
        {
            this.Genome = genome;
        }
    }
}
