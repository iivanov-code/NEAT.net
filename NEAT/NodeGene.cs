using NEAT.Abstrations;

namespace NEAT
{
    public class NodeGene : INodeGene
    {
        public NodeGene(uint innovation, NodeType type)
        {
            this.Innovation = innovation;
            this.Type = type;
        }

        public NodeType Type { get; private init; }

        public uint Innovation { get; private init; }

        public INodeGene Clone()
        {
            return new NodeGene(this.Innovation, this.Type);
        }
    }
}
