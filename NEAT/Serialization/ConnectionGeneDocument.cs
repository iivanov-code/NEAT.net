namespace NEAT.Net.Serialization
{
    internal sealed class ConnectionGeneDocument
    {
        public uint Innovation { get; set; }

        public uint From { get; set; }

        public uint To { get; set; }

        public float Weight { get; set; }

        public bool Enabled { get; set; }
    }
}
