using NEAT.Net.Abstrations.Enums;

namespace NEAT.Net.Models
{
    internal class GeneCompareType<T>
    {
        public bool Fittest { get; set; }
        public T Gene { get; set; }
        public T MatchingGene { get; set; }
        public GeneType GeneType { get; set; }
    }
}
