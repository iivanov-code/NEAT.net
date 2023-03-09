namespace NEAT.Net.Abstrations
{
    public static class DefaultConstants
    {
        public const float C1 = 1f;


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
        /// Each weight had a 90% chance of being uniformly perturbed
        /// </summary>
        public const float PERTURBED_CHANCE = 0.9f;

        /// <summary>
        /// Each weight had a 10% chance of being assigned a new random value
        /// </summary>
        public const float RANDOM_WEIGHT_CHANCE = 0.1f;

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
        /// In each generation, 25% of offspring resulted from mutation without crossover
        /// </summary>
        public const float PERCENT_OF_DIRECT_OFFSPRING = 0.25f;
    }
}
