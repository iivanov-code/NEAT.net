using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using NEAT.Net.Abstrations.Interfaces;
using NEAT.Net.Serialization;

namespace NEAT.Net
{
    /// <summary>
    /// Hands out historical markings. A connection between the same two nodes always gets the same innovation
    /// number, and splitting the same connection always creates a node with the same id, no matter which genome
    /// or generation the mutation happens in
    /// </summary>
    public class InnovationGenerator : IInnovationGenerator
    {
        private readonly Lock padlock = new Lock();
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

        internal InnovationDocument ToDocument()
        {
            lock (padlock)
            {
                return new InnovationDocument
                {
                    NextConnection = connectionCounter,
                    NextNode = nodeCounter,
                    Connections = connections.OrderBy(x => x.Value)
                                             .Select(x => new ConnectionInnovationDocument { From = x.Key.From, To = x.Key.To, Innovation = x.Value })
                                             .ToList(),
                    Splits = splitNodes.OrderBy(x => x.Key)
                                       .Select(x => new SplitInnovationDocument { Connection = x.Key, Node = x.Value })
                                       .ToList()
                };
            }
        }

        internal static InnovationGenerator FromDocument(InnovationDocument document)
        {
            var generator = new InnovationGenerator
            {
                connectionCounter = document.NextConnection,
                nodeCounter = document.NextNode
            };

            foreach (ConnectionInnovationDocument connection in document.Connections ?? new List<ConnectionInnovationDocument>())
            {
                if (connection.Innovation >= document.NextConnection || !generator.connections.TryAdd((connection.From, connection.To), connection.Innovation))
                    throw new InvalidDataException($"Invalid connection innovation {connection.Innovation}");
            }

            foreach (SplitInnovationDocument split in document.Splits ?? new List<SplitInnovationDocument>())
            {
                if (split.Node >= document.NextNode || !generator.splitNodes.TryAdd(split.Connection, split.Node))
                    throw new InvalidDataException($"Invalid node innovation {split.Node}");
            }

            return generator;
        }
    }
}
