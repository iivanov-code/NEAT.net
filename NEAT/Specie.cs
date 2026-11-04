using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using NEAT.Net.Abstrations;
using NEAT.Net.Abstrations.Interfaces;

namespace NEAT.Net
{
    public class Specie : ISpecie
    {
        public Specie(IGenome genome)
        {
            Genomes = new List<IGenome> { genome };
            Mascot = genome;
            BestFitness = float.MinValue;
        }

        /// <summary>
        /// Restores a specie from a checkpoint
        /// </summary>
        internal Specie(IGenome mascot, IEnumerable<IGenome> members, float bestFitness, int stagnation)
        {
            Genomes = new List<IGenome>(members);
            Mascot = mascot;
            BestFitness = bestFitness;
            Stagnation = stagnation;
        }

        public IGenome this[int index] => Genomes[index];

        public IList<IGenome> Genomes { get; private init; }

        public float AverageFitness => Genomes.Count > 0 ? Genomes.Average(x => x.Fitness) : 0f;

        public IGenome FittestGenome => Genomes.Count > 0 ? Genomes.MaxBy(x => x.Fitness) : null;

        public IGenome Mascot { get; private set; }

        public float BestFitness { get; private set; }

        public int Stagnation { get; private set; }

        public int Count => Genomes.Count;

        public IEnumerator<IGenome> GetEnumerator()
        {
            return Genomes.GetEnumerator();
        }

        public bool TryAddGenome(IGenome genome, float dt = DefaultConstants.DT, float c1 = DefaultConstants.C1, float c2 = DefaultConstants.C2, float c3 = DefaultConstants.C3)
        {
            if (Mascot.CompatibilityDistance(genome, c1, c2, c3) < dt)
            {
                Genomes.Add(genome);
                return true;
            }

            return false;
        }

        public void UpdateStagnation()
        {
            IGenome fittest = FittestGenome;
            if (fittest == null) return;

            if (fittest.Fitness > BestFitness)
            {
                BestFitness = fittest.Fitness;
                Stagnation = 0;
            }
            else
            {
                Stagnation++;
            }
        }

        public void StartNewGeneration(Random random)
        {
            if (Genomes.Count > 0)
            {
                Mascot = Genomes[random.Next(Genomes.Count)];
            }

            Genomes.Clear();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }
}
