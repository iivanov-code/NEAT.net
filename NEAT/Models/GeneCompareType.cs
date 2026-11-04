using NEAT.Net.Abstrations.Enums;

namespace NEAT.Net.Models
{
    /// <summary>
    /// One innovation number of two aligned genomes
    /// </summary>
    internal class GeneCompareType<T>
    {
        /// <summary>
        /// Gene of the first genome, null when only the second genome has it
        /// </summary>
        public T First { get; set; }

        /// <summary>
        /// Gene of the second genome, null when only the first genome has it
        /// </summary>
        public T Second { get; set; }

        public GeneType GeneType { get; set; }
    }
}
