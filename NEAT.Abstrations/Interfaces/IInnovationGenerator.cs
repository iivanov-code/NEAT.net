namespace NEAT.Net.Abstrations.Interfaces
{
    /// <summary>
    /// Global source of historical markings. The same structural innovation must always receive the same number,
    /// otherwise crossover cannot line up genes and the compatibility distance becomes meaningless
    /// </summary>
    public interface IInnovationGenerator
    {
        /// <summary>
        /// Innovation number of the connection between two nodes; the same pair always returns the same number
        /// </summary>
        uint GetConnectionInnovationNumber(uint fromNode, uint toNode);

        /// <summary>
        /// Id of the node created by splitting the given connection; the same connection always returns the same id
        /// </summary>
        uint GetNodeInnovationNumber(uint splitConnectionInnovation);

        /// <summary>
        /// Brand new node id that was never returned before
        /// </summary>
        uint NewNodeInnovationNumber();
    }
}
