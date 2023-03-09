using NEAT.Net.Abstrations.Enums;

namespace NEAT.Net.Abstrations.Interfaces
{
    public interface INodeGene : IHistoricalMarking, ICloneable<INodeGene>
    {
        NodeType Type { get; }
    }
}
