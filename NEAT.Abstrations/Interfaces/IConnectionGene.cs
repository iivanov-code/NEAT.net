namespace NEAT.Net.Abstrations.Interfaces
{
    public interface IConnectionGene : IHistoricalMarking, ICloneable<IConnectionGene>
    {
        INodeGene Input { get; set; }
        INodeGene Output { get; set; }
        float Weight { get; set; }
        bool Enabled { get; }
        void Disable();
        void Enable();
    }
}
