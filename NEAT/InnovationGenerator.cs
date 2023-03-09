using NEAT.Net.Abstrations.Interfaces;

namespace NEAT.Net
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
