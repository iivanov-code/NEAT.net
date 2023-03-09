using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NEAT.Net.Abstrations.Interfaces
{
    public interface IInnovationGenerator
    {
        uint GetConnectionInnovationNumber(uint fromNode, uint toNode);
    }
}
