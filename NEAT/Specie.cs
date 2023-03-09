using NEAT.Net.Abstrations;
using NEAT.Net.Abstrations.Interfaces;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NEAT.Net
{
    public class Specie : ISpecie
    {
        public Specie(IGenome genome)
        {
            Genomes = new List<IGenome>();
            Genomes.Add(genome);
            Mascot = genome;
        }

        public IGenome this[int index] => Genomes[index];

        public IList<IGenome> Genomes { get; private init; }

        public float AverageFitness
        {
            get
            {
                if (Genomes.Any())
                {
                    return Genomes.Average(x => x.Fitness);
                }
                else
                {
                    return 0;
                }
            }
        }

        public IGenome FittestGenome { get; private set; }
        public IGenome Mascot { get; private init; }

        public int Count => Genomes.Count;

        public IEnumerator<IGenome> GetEnumerator()
        {
            return Genomes.GetEnumerator();
        }

        public bool TryAddGenome(IGenome genome, float dt = DefaultConstants.DT, float c1 = DefaultConstants.C1, float c2 = DefaultConstants.C2, float c3 = DefaultConstants.C3)
        {
            if (Mascot.CompatibilityDistance(genome) < dt)
            {
                Genomes.Add(genome);
                genome.FitnessAssigned += Genome_FitnessAssigned;
                return true;
            }
            else
            {
                return false;
            }

        }

        private void Genome_FitnessAssigned(object sender, FitnessAssignedEventArgs e)
        {
            if (FittestGenome == null || e.Genome.Fitness > FittestGenome.Fitness)
            {
                FittestGenome = e.Genome;
            }
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }
}
