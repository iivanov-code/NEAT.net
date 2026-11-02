using System.Collections.Generic;
using NEAT.Net.Abstrations.Interfaces;

namespace NEAT.Net
{
    /// <summary>
    /// Hands out historical markings. A connection between the same two nodes always gets the same innovation
    /// number, and splitting the same connection always creates a node with the same id, no matter which genome
    /// or generation the mutation happens in
    /// </summary>
    public class InnovationGenerator : IInnovationGenerator
    {
        private readonly object padlock = new object();
        private readonly Dictionary<(uint From, uint To), uint> connections = new Dictionary<(uint From, uint To), uint>();
        private readonly Dictionary<uint, uint> splitNodes = new Dictionary<uint, uint>();
        private uint connectionCounter = 0;
        private uint nodeCounter = 0;

        public uint GetConnectionInnovationNumber(uint fromNode, uint toNode)
        {
            lock (padlock)
            {
                if (!connections.TryGetValue((fromNode, toNode), out uint innovation))
                {
                    innovation = connectionCounter++;
                    connections.Add((fromNode, toNode), innovation);
                }

                return innovation;
            }
        }

        public uint GetNodeInnovationNumber(uint splitConnectionInnovation)
        {
            lock (padlock)
            {
                if (!splitNodes.TryGetValue(splitConnectionInnovation, out uint node))
                {
                    node = nodeCounter++;
                    splitNodes.Add(splitConnectionInnovation, node);
                }

                return node;
            }
        }

        public uint NewNodeInnovationNumber()
        {
            lock (padlock)
            {
                return nodeCounter++;
            }
        }
    }
}
