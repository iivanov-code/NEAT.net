using NEAT.Net.Abstrations.Enums;
using NEAT.Net.Abstrations.Interfaces;

namespace NEAT.Net
{
    public class NodeGene : INodeGene
    {
        public NodeGene(uint innovation, NodeType type)
        {
            Innovation = innovation;
            Type = type;
        }

        public NodeType Type { get; private init; }

        public uint Innovation { get; private init; }

        public INodeGene Clone()
        {
            return new NodeGene(Innovation, Type);
        }
    }
}
