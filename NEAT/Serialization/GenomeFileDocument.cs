namespace NEAT.Net.Serialization
{
    internal sealed class GenomeFileDocument
    {
        public string Format { get; set; }

        public int Version { get; set; }

        public GenomeDocument Genome { get; set; }
    }
}
