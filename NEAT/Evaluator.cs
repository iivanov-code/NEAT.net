using NEAT.Net.Abstrations;
using NEAT.Net.Abstrations.Enums;
using NEAT.Net.Abstrations.Interfaces;
using System;
using System.Collections.Generic;
using System.Linq;

namespace NEAT.Net
{
    public abstract class Evaluator : IEvaluator
    {
        protected Evaluator(IGenome initialGenome, IInnovationGenerator innovationGenerator, ushort populationSize = DefaultConstants.POPULATION_SIZE)
        {
            Genomes = new List<IGenome>(populationSize);
            Species = new List<ISpecie>();

            InnovationGenerator = innovationGenerator;
            PopulationSize = populationSize;

            for (int i = 0; i < PopulationSize; i++)
            {
                Genomes.Add(initialGenome.Clone());
            }
        }

        public IGenome FittestGenome { get; private set; }
        public ICollection<IGenome> Genomes { get; private set; }
        public IInnovationGenerator InnovationGenerator { get; private init; }
        public ushort PopulationSize { get; private set; }
        public ICollection<ISpecie> Species { get; private set; }

        public void BreedGenomes()
        {
            int totalFitness = (int)Math.Ceiling(Species.Sum(x => x.AverageFitness));
            Random random = new Random();

            foreach (var specie in Species)
            {
                int count = (int)Math.Floor(specie.Count * DefaultConstants.PERCENT_OF_DIRECT_OFFSPRING);
                for (int i = 0; i < count; i++)
                {
                    IGenome genome = specie.Genomes[random.Next(0, specie.Count)];
                    specie.Genomes.Remove(genome);
                    Genomes.Add(genome);
                }
            }

            int offspringRate = totalFitness / (PopulationSize - Genomes.Count);

            foreach (var specie in Species)
            {
                int offspringsPerSpecie = (int)(specie.AverageFitness * offspringRate);

                for (int i = 0; i < offspringsPerSpecie; i++)
                {
                    IGenome parent1 = specie[random.Next(0, specie.Count)];
                    IGenome parent2 = null;

                    if (specie.Count > 1)
                    {
                        do
                        {
                            parent2 = specie[random.Next(0, specie.Count)];
                        } while (parent2 == parent1);
                    }
                    else
                    {
                        parent2 = parent1;
                    }

                    IGenome childGenome = parent1.Crossover(parent2);

                    MutateGenome(childGenome);

                    Genomes.Add(childGenome);
                }
            }

            Species.Clear();
        }

        public void CategorizeIntoSpecies()
        {
            var enumerator = Genomes.GetEnumerator();
            enumerator.MoveNext();
            Species.Add(new Specie(enumerator.Current));

            while (enumerator.MoveNext())
            {
                IGenome genome = enumerator.Current;
                if (!TryAddIntoSpecies(genome))
                {
                    Species.Add(new Specie(genome));
                }
            }

            Genomes.Clear();
        }

        public void Evaluate()
        {
            CategorizeIntoSpecies();
            EvaluateGenomes();
            PutGenomesIntoNextGeneration();
            BreedGenomes();
        }

        public abstract float EvaluateGenome(IGenome genome);

        public void EvaluateGenomes()
        {
            foreach (var species in Species)
            {
                foreach (var genome in species.Genomes)
                {
                    genome.Fitness = EvaluateGenome(genome);

                    if (FittestGenome == null || FittestGenome.Fitness < genome.Fitness)
                    {
                        FittestGenome = genome;
                    }
                }
            }
        }

        public void PutGenomesIntoNextGeneration()
        {
            foreach (var specie in Species.Where(x => x.Count > DefaultConstants.SPECIES_WITH_MORE_THAN))
            {
                specie.Genomes.Remove(specie.FittestGenome);
                Genomes.Add(specie.FittestGenome);
            }
        }

        private void MutateGenome(IGenome genome)
        {
            Random random = new Random();

            if (random.GetRandomNumber(0, 1) < DefaultConstants.CHANCE_OF_WEIGHTS_MUTATION)
            {
                genome.Mutate(MutationType.AllConnections, InnovationGenerator);
            }

            if (random.GetRandomNumber(0, 1) < DefaultConstants.PROBABILITY_OF_ADDING_NEW_LINK)
            {
                genome.Mutate(MutationType.Connection, InnovationGenerator);
            }

            if (random.GetRandomNumber(0, 1) < DefaultConstants.PROBABILITY_OF_ADDING_NEW_NODE)
            {
                genome.Mutate(MutationType.Node, InnovationGenerator);
            }
        }

        private bool TryAddIntoSpecies(IGenome genome)
        {
            foreach (var species in Species)
            {
                if (species.TryAddGenome(genome))
                {
                    return true;
                }
            }

            return false;
        }
    }
}
