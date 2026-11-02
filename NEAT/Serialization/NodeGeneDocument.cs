using NEAT.Net.Abstrations.Enums;

namespace NEAT.Net.Serialization
{
    internal sealed class NodeGeneDocument
    {
        public uint Id { get; set; }

        public NodeType Type { get; set; }
    }
}
