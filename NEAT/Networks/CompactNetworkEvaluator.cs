using ANN.Net.Abstractions.Interfaces;
using NEAT.Net.Abstrations;
using NEAT.Net.Abstrations.Interfaces;

namespace NEAT.Net.Networks
{
    /// <summary>
    /// Evaluator that scores the compact (inference-only) network of every genome. Use it instead of
    /// <see cref="NetworkEvaluator"/> when the fitness only needs the outputs of the network, which is the usual case:
    /// building and evaluating a compact network is several times faster than the trainable neuron network
    /// </summary>
    public abstract class CompactNetworkEvaluator : Evaluator
    {
        protected CompactNetworkEvaluator(IGenome initialGenome,
                                          IInnovationGenerator innovationGenerator,
                                          ushort populationSize = DefaultConstants.POPULATION_SIZE,
                                          PhenotypeSettings phenotypeSettings = null)
            : base(initialGenome, innovationGenerator, populationSize)
        {
            PhenotypeSettings = phenotypeSettings ?? new PhenotypeSettings();
        }

        protected CompactNetworkEvaluator(IGenome initialGenome,
                                          IInnovationGenerator innovationGenerator,
                                          NeatSettings settings,
                                          PhenotypeSettings phenotypeSettings = null)
            : base(initialGenome, innovationGenerator, settings)
        {
            PhenotypeSettings = phenotypeSettings ?? new PhenotypeSettings();
        }

        public PhenotypeSettings PhenotypeSettings { get; }

        public override float EvaluateGenome(IGenome genome)
        {
            return EvaluateNetwork(genome.ToCompactNetwork(PhenotypeSettings), genome);
        }

        /// <summary>
        /// Fitness of the network built from <paramref name="genome"/>; must not be negative
        /// </summary>
        protected abstract float EvaluateNetwork(ICompactNetwork network, IGenome genome);
    }
}
