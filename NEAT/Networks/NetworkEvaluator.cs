using ANN.Net.Abstractions.Interfaces;
using NEAT.Net.Abstrations;
using NEAT.Net.Abstrations.Interfaces;

namespace NEAT.Net.Networks
{
    /// <summary>
    /// Evaluator that scores the ANN.Net network of every genome instead of the genome itself
    /// </summary>
    public abstract class NetworkEvaluator : Evaluator
    {
        protected NetworkEvaluator(IGenome initialGenome,
                                   IInnovationGenerator innovationGenerator,
                                   ushort populationSize = DefaultConstants.POPULATION_SIZE,
                                   PhenotypeSettings phenotypeSettings = null)
            : base(initialGenome, innovationGenerator, populationSize)
        {
            PhenotypeSettings = phenotypeSettings ?? new PhenotypeSettings();
            MaxDegreeOfParallelism = -1;
        }

        protected NetworkEvaluator(IGenome initialGenome,
                                   IInnovationGenerator innovationGenerator,
                                   NeatSettings settings,
                                   PhenotypeSettings phenotypeSettings = null)
            : base(initialGenome, innovationGenerator, settings)
        {
            PhenotypeSettings = phenotypeSettings ?? new PhenotypeSettings();
            MaxDegreeOfParallelism = -1;
        }

        public PhenotypeSettings PhenotypeSettings { get; }

        public override float EvaluateGenome(IGenome genome)
        {
            return EvaluateNetwork(genome.ToNetwork(PhenotypeSettings), genome);
        }

        /// <summary>
        /// Fitness of the network built from <paramref name="genome"/>; must not be negative. Genomes are evaluated on
        /// every processor (see <see cref="Evaluator.MaxDegreeOfParallelism"/>), so this is called from several threads
        /// at once; every call gets its own network
        /// </summary>
        protected abstract float EvaluateNetwork(IGraphNetwork network, IGenome genome);
    }
}
