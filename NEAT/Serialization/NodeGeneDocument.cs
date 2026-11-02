using NEAT.Net.Abstrations.Enums;
using ActivationTypes = ANN.Net.Abstractions.Enums.ActivationTypes;

namespace NEAT.Net.Serialization
{
    internal sealed class NodeGeneDocument
    {
        public uint Id { get; set; }

        public NodeType Type { get; set; }

        /// <summary>
        /// Missing in version 1 files and for nodes that use the default activation of their type
        /// </summary>
        public ActivationTypes? Activation { get; set; }
    }
}
