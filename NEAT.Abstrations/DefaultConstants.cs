namespace NEAT.Net.Abstrations
{
    /// <summary>
    /// Parameters from the original NEAT paper (Stanley and Miikkulainen, 2002)
    /// </summary>
    public static class DefaultConstants
    {
        /// <summary>
        /// Weight of the excess genes in the compatibility distance
        /// </summary>
        public const float C1 = 1f;

        /// <summary>
        /// Weight of the disjoint genes in the compatibility distance
        /// </summary>
        public const float C2 = 1f;

        /// <summary>
        /// C3 Coeficient for measuring distance compatibility
        /// Default value 0.4 except for
        /// 3.0 - Double pole balancing no velocity
        /// </summary>
        public const float C3 = 0.4f;

        /// <summary>
        /// Default distance compatibility except
        /// 4.0 -  Double pole balancing no velocity
        /// </summary>
        public const float DT = 3.0f;

        /// <summary>
        /// Genomes with fewer genes than this are not normalized by their size in the compatibility distance (N = 1)
        /// </summary>
        public const int SMALL_GENOME_SIZE = 20;

        /// <summary>
        /// Default Population Size
        /// 1000 - Double pole balancing no velocity
        /// </summary>
        public const int POPULATION_SIZE = 150;

        /// <summary>
        /// The champion of each species with more than five networks
        /// was copied into the next generation unchanged
        /// </summary>
        public const int SPECIES_WITH_MORE_THAN = 5;

        /// <summary>
        /// If the maximum fitness of a species did not improve in 15 generations, the networks in the stagnant
        /// species were not allowed to reproduce
        /// </summary>
        public const int STAGNATION_LIMIT = 15;

        /// <summary>
        /// Share of the best members of every species that are allowed to become parents
        /// </summary>
        public const float SURVIVAL_THRESHOLD = 0.2f;

        /// <summary>
        /// Each weight had a 90% chance of being uniformly perturbed
        /// </summary>
        public const float PERTURBED_CHANCE = 0.9f;

        /// <summary>
        /// Each weight had a 10% chance of being assigned a new random value
        /// </summary>
        public const float RANDOM_WEIGHT_CHANCE = 0.1f;

        /// <summary>
        /// Largest change of a weight in one uniform perturbation.
        /// The weight scale suits the standard sigmoid of ANN.Net; the paper used a sigmoid about five times
        /// steeper (1 / (1 + e^(-4.9x))), which needs weights about five times smaller for the same behaviour
        /// </summary>
        public const float WEIGHT_PERTURBATION = 2.5f;

        /// <summary>
        /// New weights are drawn uniformly from [-INITIAL_WEIGHT_RANGE, INITIAL_WEIGHT_RANGE]
        /// </summary>
        public const float INITIAL_WEIGHT_RANGE = 5f;

        /// <summary>
        /// Weights are kept in [-MAX_WEIGHT, MAX_WEIGHT]
        /// </summary>
        public const float MAX_WEIGHT = 30f;

        /// <summary>
        /// There was an 80% chance of a genome having its connection weights mutated
        /// </summary>
        public const float CHANCE_OF_WEIGHTS_MUTATION = 0.8f;

        /// <summary>
        /// There was a 75% chance that an inherited gene was disabled if it was disabled in either parent
        /// </summary>
        public const float DISABLED_WEIGHT_CHANCE = 0.75f;

        /// <summary>
        /// The probability of adding a new node in smaller populations
        /// In larger populations is 0.3f
        /// </summary>
        public const float PROBABILITY_OF_ADDING_NEW_NODE = 0.03f;

        /// <summary>
        /// The probability of a new link mutation
        /// </summary>
        public const float PROBABILITY_OF_ADDING_NEW_LINK = 0.05f;

        /// <summary>
        /// Attempts to find a pair of nodes that can be linked before the link mutation gives up
        /// </summary>
        public const int ADD_LINK_ATTEMPTS = 20;

        /// <summary>
        /// In each generation, 25% of offspring resulted from mutation without crossover
        /// </summary>
        public const float PERCENT_OF_DIRECT_OFFSPRING = 0.25f;

        /// <summary>
        /// The interspecies mating rate was 0.001
        /// </summary>
        public const float INTERSPECIES_MATING_RATE = 0.001f;
    }
}
