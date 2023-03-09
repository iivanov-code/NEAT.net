using NEAT.Net.Abstrations.Enums;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NEAT.Net.Abstrations.Interfaces
{
    public interface INodeGene : IHistoricalMarking, ICloneable<INodeGene>
    {
        NodeType Type { get; }
    }
}
