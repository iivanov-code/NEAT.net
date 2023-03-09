using NEAT.Abstrations;

namespace NEAT
{
    public class ConnectionGene : IConnectionGene
    {
        public ConnectionGene(uint innovation, INodeGene input, INodeGene output, float weight)
        {
            this.Weight = weight;
            this.Innovation = innovation;
            this.Input = input;
            this.Output = output;
            this.Enabled = true;
        }

        private INodeGene input;
        public INodeGene Input
        {
            get
            {
                return input;
            }
            set
            {
                if (value == null)
                {
                    Enabled = false;
                }

                input = value;
            }
        }

        private INodeGene output;
        public INodeGene Output
        {
            get
            {
                return output;
            }
            set
            {
                if (value == null)
                {
                    Enabled = false;
                }

                output = value;
            }
        }

        public float Weight { get; set; }

        public bool Enabled { get; private set; }

        public uint Innovation { get; private set; }

        public IConnectionGene Clone()
        {
            return new ConnectionGene(Innovation, Input, Output, Weight);
        }

        public void Disable()
        {
            Enabled = false;
        }

        public void Enable()
        {
            Enabled = true;
        }
    }
}
