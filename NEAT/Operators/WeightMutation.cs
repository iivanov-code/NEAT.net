using System;
using NEAT.Net.Abstrations;
using NEAT.Net.Abstrations.Interfaces;

namespace NEAT.Net.Operators
{
    /// <summary>
    /// Changes every weight: with <see cref="NeatSettings.WeightPerturbationChance"/> by a uniform perturbation of up to
    /// <see cref="NeatSettings.WeightPerturbation"/> (kept within <see cref="NeatSettings.MaxWeight"/>), otherwise by a
    /// new random value. Applied with <see cref="NeatSettings.WeightMutationRate"/>
    /// </summary>
    public sealed class WeightMutation : IMutation
    {
        public float GetProbability(EvolutionContext context) => context.Settings.WeightMutationRate;

        public void Apply(IGenome genome, EvolutionContext context)
        {
            NeatSettings settings = context.Settings;
            Random random = context.Random;

            foreach (IConnectionGene connection in genome.Connections.Values)
            {
                if (random.NextDouble() < settings.WeightPerturbationChance)
                {
                    float weight = connection.Weight + random.NextFloat(-settings.WeightPerturbation, settings.WeightPerturbation);
                    connection.Weight = Math.Clamp(weight, -settings.MaxWeight, settings.MaxWeight);
                }
                else
                {
                    connection.Weight = GenomeOperations.RandomWeight(settings, random);
                }
            }
        }
    }
}
