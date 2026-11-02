using NEAT.Net.Abstrations.Enums;
using ActivationTypes = ANN.Net.Abstractions.Enums.ActivationTypes;

namespace NEAT.Net.Abstrations.Interfaces
{
    public interface INodeGene : IHistoricalMarking, ICloneable<INodeGene>
    {
        NodeType Type { get; }

        /// <summary>
        /// Activation of the node in the network. Null (input and bias nodes, and genomes saved before
        /// activations were part of genes) means the default activation of the phenotype settings for the node type
        /// </summary>
        ActivationTypes? Activation { get; set; }
    }
}
