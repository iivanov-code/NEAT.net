namespace NEAT.Net.Abstrations.Interfaces
{
    public interface IInnovationGenerator
    {
        uint GetConnectionInnovationNumber(uint fromNode, uint toNode);
    }
}
