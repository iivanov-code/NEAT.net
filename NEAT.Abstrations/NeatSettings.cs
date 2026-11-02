using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Enums;

namespace NEAT.Net.Abstrations
{
    /// <summary>
    /// Parameters of a NEAT run. The defaults are the values of <see cref="DefaultConstants"/>, which follow the
    /// original NEAT paper with weights scaled for the standard sigmoid of ANN.Net. Settings can be changed between
    /// generations; <see cref="Validate"/> is called before every generation
    /// </summary>
    public class NeatSettings
    {
        // ---- population and species ----

        public ushort PopulationSize { get; set; } = DefaultConstants.POPULATION_SIZE;

        /// <summary>
        /// Genomes closer than this to the representative of a specie belong to that specie (δt)
        /// </summary>
        public float CompatibilityThreshold { get; set; } = DefaultConstants.DT;

        /// <summary>
        /// Weight of the excess genes in the compatibility distance (c1)
        /// </summary>
        public float ExcessCoefficient { get; set; } = DefaultConstants.C1;

        /// <summary>
        /// Weight of the disjoint genes in the compatibility distance (c2)
        /// </summary>
        public float DisjointCoefficient { get; set; } = DefaultConstants.C2;

        /// <summary>
        /// Weight of the average weight difference of matching genes in the compatibility distance (c3)
        /// </summary>
        public float WeightDifferenceCoefficient { get; set; } = DefaultConstants.C3;

        /// <summary>
        /// Weight of the matching node genes with different activations in the compatibility distance (c4).
        /// 0 leaves activations out of speciation
        /// </summary>
        public float ActivationDifferenceCoefficient { get; set; }

        /// <summary>
        /// Genomes with fewer genes than this are not normalized by their size in the compatibility distance (N = 1)
        /// </summary>
        public int SmallGenomeSize { get; set; } = DefaultConstants.SMALL_GENOME_SIZE;

        /// <summary>
        /// The champion of every specie with more members than this is copied into the next generation unchanged
        /// </summary>
        public int ElitismMinimumSpecieSize { get; set; } = DefaultConstants.SPECIES_WITH_MORE_THAN;

        /// <summary>
        /// Species whose best fitness did not improve for this many generations get no offspring,
        /// unless they hold the best genome
        /// </summary>
        public int StagnationLimit { get; set; } = DefaultConstants.STAGNATION_LIMIT;

        /// <summary>
        /// Share of the best members of every specie that are allowed to become parents
        /// </summary>
        public float SurvivalThreshold { get; set; } = DefaultConstants.SURVIVAL_THRESHOLD;

        // ---- reproduction ----

        /// <summary>
        /// Share of the offspring that are mutated copies instead of crossover children
        /// </summary>
        public float MutationOnlyRate { get; set; } = DefaultConstants.PERCENT_OF_DIRECT_OFFSPRING;

        /// <summary>
        /// Chance that the second parent of a crossover comes from another specie
        /// </summary>
        public float InterspeciesMatingRate { get; set; } = DefaultConstants.INTERSPECIES_MATING_RATE;

        /// <summary>
        /// Chance that a gene disabled in either parent is disabled in the child
        /// </summary>
        public float DisableInheritedChance { get; set; } = DefaultConstants.DISABLED_WEIGHT_CHANCE;

        // ---- weights ----

        /// <summary>
        /// Chance that the weights of an offspring are mutated
        /// </summary>
        public float WeightMutationRate { get; set; } = DefaultConstants.CHANCE_OF_WEIGHTS_MUTATION;

        /// <summary>
        /// Chance that a mutated weight is perturbed rather than replaced with a new random value
        /// </summary>
        public float WeightPerturbationChance { get; set; } = DefaultConstants.PERTURBED_CHANCE;

        /// <summary>
        /// Largest change of a weight in one uniform perturbation
        /// </summary>
        public float WeightPerturbation { get; set; } = DefaultConstants.WEIGHT_PERTURBATION;

        /// <summary>
        /// New weights are drawn uniformly from [-InitialWeightRange, InitialWeightRange]
        /// </summary>
        public float InitialWeightRange { get; set; } = DefaultConstants.INITIAL_WEIGHT_RANGE;

        /// <summary>
        /// Weights are kept in [-MaxWeight, MaxWeight]
        /// </summary>
        public float MaxWeight { get; set; } = DefaultConstants.MAX_WEIGHT;

        // ---- structure ----

        /// <summary>
        /// Chance that an offspring gets a new node
        /// </summary>
        public float AddNodeRate { get; set; } = DefaultConstants.PROBABILITY_OF_ADDING_NEW_NODE;

        /// <summary>
        /// Chance that an offspring gets a new connection
        /// </summary>
        public float AddConnectionRate { get; set; } = DefaultConstants.PROBABILITY_OF_ADDING_NEW_LINK;

        /// <summary>
        /// Attempts to find a pair of nodes that can be linked before the connection mutation gives up
        /// </summary>
        public int AddConnectionAttempts { get; set; } = DefaultConstants.ADD_LINK_ATTEMPTS;

        // ---- activations ----

        /// <summary>
        /// Activations new hidden nodes can get and activation mutations can choose from
        /// </summary>
        public IList<ActivationTypes> HiddenActivations { get; set; } = new List<ActivationTypes> { ActivationTypes.Sigmoid };

        /// <summary>
        /// Activations of output nodes; they evolve only when more than one is listed
        /// </summary>
        public IList<ActivationTypes> OutputActivations { get; set; } = new List<ActivationTypes> { ActivationTypes.Sigmoid };

        /// <summary>
        /// Chance that an offspring gets one node with a different activation
        /// (only when there is more than one activation to choose from)
        /// </summary>
        public float ActivationMutationRate { get; set; } = 0.1f;

        /// <summary>
        /// Copy that can be changed without affecting this instance
        /// </summary>
        public NeatSettings Clone()
        {
            var clone = (NeatSettings)MemberwiseClone();
            clone.HiddenActivations = new List<ActivationTypes>(HiddenActivations ?? Enumerable.Empty<ActivationTypes>());
            clone.OutputActivations = new List<ActivationTypes>(OutputActivations ?? Enumerable.Empty<ActivationTypes>());
            return clone;
        }

        /// <summary>
        /// Throws when a parameter is out of range
        /// </summary>
        public void Validate()
        {
            Check(PopulationSize > 0, PopulationSize, nameof(PopulationSize), "The population needs at least one genome");

            NotNegative(CompatibilityThreshold, nameof(CompatibilityThreshold));
            NotNegative(ExcessCoefficient, nameof(ExcessCoefficient));
            NotNegative(DisjointCoefficient, nameof(DisjointCoefficient));
            NotNegative(WeightDifferenceCoefficient, nameof(WeightDifferenceCoefficient));
            NotNegative(ActivationDifferenceCoefficient, nameof(ActivationDifferenceCoefficient));
            NotNegative(SmallGenomeSize, nameof(SmallGenomeSize));
            NotNegative(ElitismMinimumSpecieSize, nameof(ElitismMinimumSpecieSize));
            NotNegative(WeightPerturbation, nameof(WeightPerturbation));
            NotNegative(InitialWeightRange, nameof(InitialWeightRange));

            Check(StagnationLimit > 0, StagnationLimit, nameof(StagnationLimit), "Must be positive");
            Check(AddConnectionAttempts > 0, AddConnectionAttempts, nameof(AddConnectionAttempts), "Must be positive");
            Check(MaxWeight > 0, MaxWeight, nameof(MaxWeight), "Must be positive");
            Check(SurvivalThreshold > 0 && SurvivalThreshold <= 1, SurvivalThreshold, nameof(SurvivalThreshold), "Must be in (0, 1]");

            Probability(MutationOnlyRate, nameof(MutationOnlyRate));
            Probability(InterspeciesMatingRate, nameof(InterspeciesMatingRate));
            Probability(DisableInheritedChance, nameof(DisableInheritedChance));
            Probability(WeightMutationRate, nameof(WeightMutationRate));
            Probability(WeightPerturbationChance, nameof(WeightPerturbationChance));
            Probability(AddNodeRate, nameof(AddNodeRate));
            Probability(AddConnectionRate, nameof(AddConnectionRate));
            Probability(ActivationMutationRate, nameof(ActivationMutationRate));

            Activations(HiddenActivations, nameof(HiddenActivations));
            Activations(OutputActivations, nameof(OutputActivations));
        }

        private static void NotNegative(float value, string name)
        {
            Check(value >= 0, value, name, "Must not be negative");
        }

        private static void Probability(float value, string name)
        {
            Check(value >= 0 && value <= 1, value, name, "Must be a probability in [0, 1]");
        }

        /// <summary>
        /// Throws <see cref="ArgumentOutOfRangeException"/> naming the setting when <paramref name="condition"/> does not hold
        /// (also for NaN, since every comparison with NaN is false)
        /// </summary>
        private static void Check(bool condition, object value, string name, string message)
        {
            if (!condition) throw new ArgumentOutOfRangeException(name, value, message);
        }

        private static void Activations(IList<ActivationTypes> activations, string name)
        {
            if (activations == null || activations.Count == 0)
                throw new ArgumentException("At least one activation is required", name);

            foreach (ActivationTypes activation in activations)
            {
                if (!Enum.IsDefined(activation) || activation == ActivationTypes.BReLU)
                    throw new ArgumentException($"{activation} cannot be evolved", name);
            }
        }
    }
}
