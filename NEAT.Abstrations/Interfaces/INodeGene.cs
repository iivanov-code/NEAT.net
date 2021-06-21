using NEAT.Abstrations.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NEAT.Abstrations
{
    public interface INodeGene : IHistoricalMarking, ICloneable<INodeGene>
    {
        NodeType Type { get; }
    }
}
