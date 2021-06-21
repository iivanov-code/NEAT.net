using NEAT.Abstrations;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NEAT
{
    public class InnovationGenerator : IInnovationGenerator
    {
        private uint counter = 0;
        private object padlock = new object();
        public uint GetConnectionInnovationNumber(uint fromNode, uint toNode)
        {
            uint tempCounter = counter;
            lock (padlock)
            {
                counter++;
            }

            return tempCounter;
        }
    }
}
