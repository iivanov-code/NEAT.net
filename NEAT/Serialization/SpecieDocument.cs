using System.Collections.Generic;

namespace NEAT.Net.Serialization
{
    internal sealed class SpecieDocument
    {
        public int Mascot { get; set; }

        public List<int> Members { get; set; } = new List<int>();

        public float BestFitness { get; set; }

        public int Stagnation { get; set; }
    }
}
