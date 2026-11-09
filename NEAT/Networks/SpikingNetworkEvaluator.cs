using ANN.Net.Abstractions.Spiking;
using NEAT.Net.Abstrations;
using NEAT.Net.Abstrations.Interfaces;

namespace NEAT.Net.Networks
{
    /// <summary>
    /// Evaluator that scores the spiking network of every genome: evolution then finds topologies and weights that
    /// work with spikes directly, without training an analog network first. With
    /// <see cref="SpikingPhenotypeSettings.Learn"/> a network can also learn with STDP during its evaluation, and
    /// <see cref="SpikingGenomeExtensions.ApplyWeights"/> writes what it
    /// learned back into the genome
    /// </summary>
    public abstract class SpikingNetworkEvaluator : Evaluator
    {
        protected SpikingNetworkEvaluator(IGenome initialGenome,
                                          IInnovationGenerator innovationGenerator,
                                          ushort populationSize = DefaultConstants.POPULATION_SIZE,
                                          SpikingPhenotypeSettings phenotypeSettings = null)
            : base(initialGenome, innovationGenerator, populationSize)
        {
            PhenotypeSettings = phenotypeSettings ?? new SpikingPhenotypeSettings();
            MaxDegreeOfParallelism = -1;
        }

        protected SpikingNetworkEvaluator(IGenome initialGenome,
                                          IInnovationGenerator innovationGenerator,
                                          NeatSettings settings,
                                          SpikingPhenotypeSettings phenotypeSettings = null)
            : base(initialGenome, innovationGenerator, settings)
        {
            PhenotypeSettings = phenotypeSettings ?? new SpikingPhenotypeSettings();
            MaxDegreeOfParallelism = -1;
        }

        public SpikingPhenotypeSettings PhenotypeSettings { get; }

        public override float EvaluateGenome(IGenome genome)
        {
            return EvaluateNetwork(genome.ToSpikingNetwork(PhenotypeSettings), genome);
        }

        /// <summary>
        /// Fitness of the spiking network built from <paramref name="genome"/>; must not be negative. Genomes are
        /// evaluated on every processor (see <see cref="Evaluator.MaxDegreeOfParallelism"/>), so this is called from
        /// several threads at once; every call gets its own network
        /// </summary>
        protected abstract float EvaluateNetwork(ISpikingNetwork network, IGenome genome);
    }
}
