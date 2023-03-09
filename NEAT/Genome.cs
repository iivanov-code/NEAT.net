using System;
using System.Collections.Generic;
using System.Linq;
using NEAT.Net.Abstrations;
using NEAT.Net.Abstrations.Enums;
using NEAT.Net.Abstrations.Interfaces;
using NEAT.Net.Models;

namespace NEAT.Net
{
    public class Genome : IGenome
    {
        public Genome()
        {
            ID = NEATUtils.NewID();
            Nodes = new Dictionary<uint, INodeGene>();
            Connections = new Dictionary<uint, IConnectionGene>();
        }


        private event EventHandler<FitnessAssignedEventArgs> fitnessAssigned;
        public event EventHandler<FitnessAssignedEventArgs> FitnessAssigned
        {
            add
            {
                fitnessAssigned += value;
            }
            remove
            {
                fitnessAssigned -= value;
            }
        }

        public IDictionary<uint, IConnectionGene> Connections { get; private set; }

        private float fitness;
        public float Fitness
        {
            get
            {
                return fitness;
            }
            set
            {
                fitness = value;
                OnFitnessAssinged();
            }
        }

        public uint ID { get; private init; }
        public IDictionary<uint, INodeGene> Nodes { get; private set; }
        public IGenome Clone()
        {
            Genome genome = new Genome();

            foreach (var connection in Connections.Values)
            {
                genome.Connections.Add(connection.Innovation, connection.Clone());
            }

            foreach (var node in Nodes.Values)
            {
                genome.Nodes.Add(node.Innovation, node.Clone());
            }

            return genome;
        }

        public int CompareTo(IGenome other)
        {
            float distance = Fitness - other.Fitness;

            if (distance > 0)
            {
                return 1; //this genome is more fit than the other
            }
            else if (distance < 0)
            {
                return -1; //this genome is less fit than the other
            }
            else
            {
                return 0; //Both genomes are equal
            }
        }

        /// <summary>
        /// C1*E + C2*D + C3*W
        ///  N      N
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        public float CompatibilityDistance(IGenome other, float c1 = DefaultConstants.C1, float c2 = DefaultConstants.C2, float c3 = DefaultConstants.C3)
        {
            int matchingGenes = 0, disjointGenes = 0, excessGenes = 0;

            int numberGenes = Math.Max(Nodes.Count + Connections.Count, other.Connections.Count + Connections.Count);

            if (numberGenes < 20) numberGenes = 1;

            float avgWeightDistance = 0f;

            foreach (var geneDesrc in GetGeneTypes(Connections, other.Connections))
            {
                switch (geneDesrc.GeneType)
                {
                    case GeneType.Matching:
                        matchingGenes++;
                        avgWeightDistance += Math.Abs(geneDesrc.Gene.Weight - geneDesrc.MatchingGene.Weight);
                        break;
                    case GeneType.Disjoint:
                        disjointGenes++;
                        break;
                    case GeneType.Excess:
                        excessGenes++;
                        break;
                }
            }

            avgWeightDistance = avgWeightDistance / matchingGenes;

            foreach (var gene in GetGeneTypes(Nodes, other.Nodes))
            {
                switch (gene.GeneType)
                {
                    case GeneType.Matching:
                        matchingGenes++;
                        break;
                    case GeneType.Disjoint:
                        disjointGenes++;
                        break;
                    case GeneType.Excess:
                        excessGenes++;
                        break;
                }
            }

            return c1 * excessGenes / numberGenes + c2 * disjointGenes / numberGenes + c3 * avgWeightDistance;
        }

        public IGenome Crossover(IGenome other)
        {
            Genome childGenome = new Genome();

            IGenome fittest = Fitness > other.Fitness ? this : other;
            other = Fitness < other.Fitness ? this : other;

            foreach (var node in fittest.Nodes)
            {
                childGenome.Nodes.Add(node.Key, node.Value.Clone());
            }

            Random random = new Random();

            foreach (var gene in GetGeneTypes(fittest.Connections, other.Connections).Where(x => x.Fittest))
            {
                var connection = gene.Gene.Clone();
                connection.Input = childGenome.Nodes.GetValueOrDefault(connection.Input.Innovation);
                connection.Output = childGenome.Nodes.GetValueOrDefault(connection.Output.Innovation);

                if (gene.GeneType == GeneType.Matching && !gene.Gene.Enabled && !gene.MatchingGene.Enabled)
                {
                    if (random.GetRandomNumber(0, 1) > DefaultConstants.DISABLED_WEIGHT_CHANCE)
                    {
                        connection.Enable();
                    }
                }

                childGenome.Connections.Add(gene.Gene.Innovation, connection);
            }

            return childGenome;
        }

        public void Mutate(MutationType mutationType, IInnovationGenerator generator)
        {
            switch (mutationType)
            {
                case MutationType.Connection:
                    AddConnectionMutation(generator);
                    break;
                case MutationType.Node:
                    AddNodeMutation(generator);
                    break;
                case MutationType.AllConnections:
                    MutateAllConnections();
                    break;
                default:
                    throw new ArgumentException(nameof(mutationType));
            }
        }

        private static IEnumerable<GeneCompareType<TValue>> GetGeneTypes<TValue>(IDictionary<uint, TValue> fittest, IDictionary<uint, TValue> other)
            where TValue : IHistoricalMarking
        {
            uint firstMax = fittest.Select(x => x.Key).Max();
            uint secondMax = other.Select(x => x.Key).Max();

            uint max = Math.Max(firstMax, secondMax);
            uint min = Math.Min(fittest.Select(x => x.Key).Min(), other.Select(x => x.Key).Min());

            uint minMax = Math.Min(firstMax, secondMax);

            Random random = new Random();

            for (uint i = min; i < max; i++)
            {
                TValue firstGene = fittest.GetValueOrDefault(i);
                TValue secondGene = other.GetValueOrDefault(i);

                if (firstGene != null || secondGene != null)
                {
                    if (firstGene != null && secondGene != null)
                    {
                        var geneDescr = new GeneCompareType<TValue>
                        {
                            GeneType = GeneType.Matching,
                            Fittest = true,
                        };

                        if (random.GetRandomBool())
                        {
                            geneDescr.Gene = firstGene;
                            geneDescr.MatchingGene = secondGene;
                        }
                        else
                        {
                            geneDescr.Gene = secondGene;
                            geneDescr.MatchingGene = firstGene;
                        }

                        yield return geneDescr;
                    }
                    else if (firstGene != null && i < minMax) //fittest genome disjoint gene
                    {
                        yield return new GeneCompareType<TValue>
                        {
                            GeneType = GeneType.Disjoint,
                            Gene = firstGene,
                            Fittest = true,
                        };
                    }
                    else if (secondGene != null && i < minMax) //less fit genome disjoint gene
                    {
                        yield return new GeneCompareType<TValue>
                        {
                            GeneType = GeneType.Disjoint,
                            Gene = secondGene,
                            Fittest = false,
                        };
                    }
                    else if (firstGene != null && i > minMax) //fittest genome excess gene
                    {
                        yield return new GeneCompareType<TValue>
                        {
                            GeneType = GeneType.Excess,
                            Gene = firstGene,
                            Fittest = true
                        };
                    }
                    else if (secondGene != null && i > minMax) //less fit genome excess gene
                    {
                        yield return new GeneCompareType<TValue>
                        {
                            GeneType = GeneType.Excess,
                            Gene = secondGene,
                            Fittest = false
                        };
                    }
                }
            }
        }

        private void AddConnectionMutation(IInnovationGenerator generator)
        {
            Random rand = new Random();
            INodeGene fromNode = Nodes.GetElementAt(rand.Next(0, Nodes.Count));
            INodeGene toNode = null;

            do
            {
                toNode = Nodes.GetElementAt(rand.Next(0, Nodes.Count));
            } while (fromNode == toNode || fromNode.Type == toNode.Type);

            bool hasConnection = Connections.Where(x => x.Value.Input.Innovation == fromNode.Innovation && x.Value.Output.Innovation == toNode.Innovation
              || x.Value.Input.Innovation == toNode.Innovation && x.Value.Output.Innovation == fromNode.Innovation).Any();

            if (!hasConnection)
            {
                if (fromNode.Type > toNode.Type)
                {
                    NEATUtils.Swap(ref fromNode, ref toNode);
                }

                float weight = new Random().GetRandomNumber(0, 1);
                IConnectionGene connection = new ConnectionGene(generator.GetConnectionInnovationNumber(fromNode.Innovation, toNode.Innovation), fromNode, toNode, weight);
                Connections.Add(connection.Innovation, connection);
            }
        }

        private void AddNodeMutation(IInnovationGenerator generator)
        {
            INodeGene newNode = new NodeGene((uint)Nodes.Count, NodeType.Hidden);
            var random = new Random();
            int index = random.Next(0, Connections.Count);
            IConnectionGene connection = Connections.GetElementAt(index);

            connection.Disable();

            uint innovation = generator.GetConnectionInnovationNumber(connection.Input.Innovation, newNode.Innovation);
            IConnectionGene fromConnection = new ConnectionGene(innovation, connection.Input, newNode, 1f);
            Connections.Add(fromConnection.Innovation, fromConnection);

            innovation = generator.GetConnectionInnovationNumber(newNode.Innovation, connection.Output.Innovation);
            IConnectionGene toConnection = new ConnectionGene(innovation, newNode, connection.Output, random.GetRandomNumber(0, 1));
            Connections.Add(toConnection.Innovation, toConnection);

            Nodes.Add(newNode.Innovation, newNode);
        }

        private void OnFitnessAssinged()
        {
            if (fitnessAssigned != null)
            {
                fitnessAssigned(this, new FitnessAssignedEventArgs(this));
            }
        }

        private void MutateAllConnections()
        {
            Random random = new Random();
            foreach (var connection in Connections.Select(x => x.Value))
            {
                if (random.GetRandomNumber(0, 1) < DefaultConstants.PERTURBED_CHANCE)
                {
                    float distance = 1 - Math.Abs(connection.Weight);
                    connection.Weight += random.GetRandomNumber(-1 * distance, distance);
                }
                else
                {
                    connection.Weight = random.GetRandomNumber(0, 1);
                }
            }
        }
    }
}
