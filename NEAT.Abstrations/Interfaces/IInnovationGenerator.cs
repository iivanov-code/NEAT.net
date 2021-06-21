﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NEAT.Abstrations
{
    public interface IInnovationGenerator
    {
        uint GetConnectionInnovationNumber(uint fromNode, uint toNode);
    }
}
