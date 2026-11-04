using System.IO;
using System.Text;
using NEAT.Net.Abstrations.Interfaces;

namespace NEAT.Net.Tests
{
    /// <summary>
    /// Genome as saved JSON, to compare genomes gene by gene
    /// </summary>
    internal static class GenomeJson
    {
        public static string Of(IGenome genome)
        {
            using var stream = new MemoryStream();
            GenomeSerializer.Save(genome, stream);
            return Encoding.UTF8.GetString(stream.ToArray());
        }
    }
}
