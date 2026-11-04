using NEAT.Net.Abstrations.Enums;
using NEAT.Net.Abstrations.Interfaces;
using ActivationTypes = ANN.Net.Abstractions.Enums.ActivationTypes;

namespace NEAT.Net
{
    public class NodeGene : INodeGene
    {
        public NodeGene(uint innovation, NodeType type, ActivationTypes? activation = null)
        {
            Innovation = innovation;
            Type = type;
            Activation = activation;
        }

        public NodeType Type { get; private init; }

        public uint Innovation { get; private init; }

        public ActivationTypes? Activation { get; set; }

        public INodeGene Clone()
        {
            return new NodeGene(Innovation, Type, Activation);
        }
    }
}
