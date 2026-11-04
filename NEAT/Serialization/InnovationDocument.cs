using System.Collections.Generic;

namespace NEAT.Net.Serialization
{
    internal sealed class InnovationDocument
    {
        public uint NextConnection { get; set; }

        public uint NextNode { get; set; }

        public List<ConnectionInnovationDocument> Connections { get; set; } = new List<ConnectionInnovationDocument>();

        public List<SplitInnovationDocument> Splits { get; set; } = new List<SplitInnovationDocument>();
    }
}
