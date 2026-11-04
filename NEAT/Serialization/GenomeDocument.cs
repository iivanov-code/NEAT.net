using System.Collections.Generic;

namespace NEAT.Net.Serialization
{
    internal sealed class GenomeDocument
    {
        public float Fitness { get; set; }

        public List<NodeGeneDocument> Nodes { get; set; } = new List<NodeGeneDocument>();

        public List<ConnectionGeneDocument> Connections { get; set; } = new List<ConnectionGeneDocument>();
    }
}
