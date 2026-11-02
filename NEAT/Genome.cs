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

        /// <summary>
        /// Minimal starting genome: every input (and the bias) connected directly to every output, without hidden nodes
        /// </summary>
        public static Genome CreateMinimal(int inputsCount, int outputsCount, IInnovationGenerator generator, bool withBias = true, NeatSettings settings = null)
        {
            settings ??= DefaultSettings;
            if (inputsCount <= 0) throw new ArgumentOutOfRangeException(nameof(inputsCount));
            if (outputsCount <= 0) throw new ArgumentOutOfRangeException(nameof(outputsCount));
            if (generator == null) throw new ArgumentNullException(nameof(generator));

            Genome genome = new Genome();
            var sources = new List<INodeGene>();
            var outputs = new List<INodeGene>();

            for (int i = 0; i < inputsCount; i++)
            {
                sources.Add(genome.AddNode(generator.NewNodeInnovationNumber(), NodeType.Input));
            }

            if (withBias)
            {
                sources.Add(genome.AddNode(generator.NewNodeInnovationNumber(), NodeType.Bias));
            }

            for (int i = 0; i < outputsCount; i++)
            {
                outputs.Add(genome.AddNode(generator.NewNodeInnovationNumber(), NodeType.Output));
            }

            foreach (INodeGene from in sources)
            {
                foreach (INodeGene to in outputs)
                {
                    uint innovation = generator.GetConnectionInnovationNumber(from.Innovation, to.Innovation);
                    genome.Connections.Add(innovation, new ConnectionGene(innovation, from, to, RandomWeight(settings)));
                }
            }

            return genome;
        }

        /// <summary>
        /// Used by the overloads without settings
        /// </summary>
        private static readonly NeatSettings DefaultSettings = new NeatSettings();

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

        /// <summary>
        /// Deep copy with a new <see cref="ID"/>; the connections of the copy point to the nodes of the copy
        /// </summary>
        public IGenome Clone()
        {
            Genome genome = new Genome();
            genome.fitness = fitness;

            foreach (var node in Nodes.Values)
            {
                genome.Nodes.Add(node.Innovation, node.Clone());
            }

            foreach (var connection in Connections.Values)
            {
                genome.Connections.Add(connection.Innovation, genome.CopyConnection(connection));
            }

            return genome;
        }

        public int CompareTo(IGenome other)
        {
            return Fitness.CompareTo(other.Fitness);
        }

        /// <summary>
        /// δ = c1*E/N + c2*D/N + c3*W, where E and D are the excess and disjoint connection genes, W the average
        /// weight difference of the matching genes and N the number of connection genes in the larger genome
        /// (1 for small genomes)
        /// </summary>
        public float CompatibilityDistance(IGenome other, float c1 = DefaultConstants.C1, float c2 = DefaultConstants.C2, float c3 = DefaultConstants.C3)
        {
            return CompatibilityDistance(other, c1, c2, c3, DefaultConstants.SMALL_GENOME_SIZE);
        }

        public float CompatibilityDistance(IGenome other, NeatSettings settings)
        {
            ArgumentNullException.ThrowIfNull(settings);
            return CompatibilityDistance(other, settings.ExcessCoefficient, settings.DisjointCoefficient, settings.WeightDifferenceCoefficient, settings.SmallGenomeSize);
        }

        private float CompatibilityDistance(IGenome other, float c1, float c2, float c3, int smallGenomeSize)
        {
            if (other == null) throw new ArgumentNullException(nameof(other));

            int matchingGenes = 0, disjointGenes = 0, excessGenes = 0;
            float weightDifference = 0f;

            foreach (var gene in AlignGenes(Connections, other.Connections))
            {
                switch (gene.GeneType)
                {
                    case GeneType.Matching:
                        matchingGenes++;
                        weightDifference += Math.Abs(gene.First.Weight - gene.Second.Weight);
                        break;
                    case GeneType.Disjoint:
                        disjointGenes++;
                        break;
                    case GeneType.Excess:
                        excessGenes++;
                        break;
                }
            }

            int numberGenes = Math.Max(Connections.Count, other.Connections.Count);
            if (numberGenes < smallGenomeSize) numberGenes = 1;

            float averageWeightDifference = matchingGenes > 0 ? weightDifference / matchingGenes : 0f;

            return (c1 * excessGenes / numberGenes) + (c2 * disjointGenes / numberGenes) + (c3 * averageWeightDifference);
        }

        /// <summary>
        /// Matching genes are inherited randomly from either parent, disjoint and excess genes from the fitter parent
        /// (from a random one when both are equally fit). The child has the topology of that parent, so it stays acyclic
        /// </summary>
        public IGenome Crossover(IGenome other)
        {
            return Crossover(other, DefaultSettings);
        }

        /// <inheritdoc cref="Crossover(IGenome)"/>
        public IGenome Crossover(IGenome other, NeatSettings settings)
        {
            if (other == null) throw new ArgumentNullException(nameof(other));
            ArgumentNullException.ThrowIfNull(settings);

            Random random = NEATUtils.Random;
            IGenome fittest, weakest;

            int comparison = Fitness.CompareTo(other.Fitness);
            if (comparison > 0 || (comparison == 0 && random.GetRandomBool()))
            {
                fittest = this;
                weakest = other;
            }
            else
            {
                fittest = other;
                weakest = this;
            }

            Genome childGenome = new Genome();

            foreach (var node in fittest.Nodes.Values)
            {
                childGenome.Nodes.Add(node.Innovation, node.Clone());
            }

            foreach (var gene in AlignGenes(fittest.Connections, weakest.Connections))
            {
                if (gene.First == null)
                {
                    continue; // disjoint and excess genes of the weaker parent are not inherited
                }

                IConnectionGene connection;
                if (gene.GeneType == GeneType.Matching)
                {
                    IConnectionGene source = random.GetRandomBool() ? gene.First : gene.Second;
                    connection = childGenome.CopyConnection(source);

                    if (!gene.First.Enabled || !gene.Second.Enabled)
                    {
                        if (random.NextDouble() < settings.DisableInheritedChance)
                        {
                            connection.Disable();
                        }
                        else
                        {
                            connection.Enable();
                        }
                    }
                }
                else
                {
                    connection = childGenome.CopyConnection(gene.First);
                }

                childGenome.Connections.Add(connection.Innovation, connection);
            }

            return childGenome;
        }

        public void Mutate(MutationType mutationType, IInnovationGenerator generator)
        {
            Mutate(mutationType, generator, DefaultSettings);
        }

        public void Mutate(MutationType mutationType, IInnovationGenerator generator, NeatSettings settings)
        {
            ArgumentNullException.ThrowIfNull(settings);

            switch (mutationType)
            {
                case MutationType.Connection:
                    AddConnectionMutation(generator, settings);
                    break;
                case MutationType.Node:
                    AddNodeMutation(generator);
                    break;
                case MutationType.AllConnections:
                    MutateAllConnections(settings);
                    break;
                default:
                    throw new ArgumentException("Unknown mutation type", nameof(mutationType));
            }
        }

        /// <summary>
        /// Lines up the genes of two genomes by innovation number. Genes of one genome that fall inside the innovation
        /// range of the other are disjoint, genes beyond the other genome's highest innovation are excess
        /// </summary>
        private static IEnumerable<GeneCompareType<TValue>> AlignGenes<TValue>(IDictionary<uint, TValue> first, IDictionary<uint, TValue> second)
            where TValue : class, IHistoricalMarking
        {
            long firstMax = first.Count > 0 ? first.Keys.Max() : -1;
            long secondMax = second.Count > 0 ? second.Keys.Max() : -1;

            foreach (uint innovation in first.Keys.Union(second.Keys).OrderBy(x => x))
            {
                TValue firstGene = first.GetValueOrDefault(innovation);
                TValue secondGene = second.GetValueOrDefault(innovation);

                GeneType type;
                if (firstGene != null && secondGene != null)
                {
                    type = GeneType.Matching;
                }
                else if (firstGene != null)
                {
                    type = innovation > secondMax ? GeneType.Excess : GeneType.Disjoint;
                }
                else
                {
                    type = innovation > firstMax ? GeneType.Excess : GeneType.Disjoint;
                }

                yield return new GeneCompareType<TValue>
                {
                    First = firstGene,
                    Second = secondGene,
                    GeneType = type
                };
            }
        }

        private static float RandomWeight(NeatSettings settings)
        {
            return NEATUtils.Random.NextFloat(-settings.InitialWeightRange, settings.InitialWeightRange);
        }

        private INodeGene AddNode(uint innovation, NodeType type)
        {
            INodeGene node = new NodeGene(innovation, type);
            Nodes.Add(innovation, node);
            return node;
        }

        /// <summary>
        /// Copies a connection gene of any genome and points it to the nodes of this genome
        /// </summary>
        private IConnectionGene CopyConnection(IConnectionGene connection)
        {
            IConnectionGene copy = connection.Clone();
            copy.Input = Nodes[connection.Input.Innovation];
            copy.Output = Nodes[connection.Output.Innovation];
            return copy;
        }

        private static bool CanBeSource(INodeGene node) => node.Type != NodeType.Output;

        private static bool CanBeTarget(INodeGene node) => node.Type == NodeType.Hidden || node.Type == NodeType.Output;

        /// <summary>
        /// Adds a connection between two unconnected nodes. Only feed-forward connections are created: the new
        /// connection may not close a cycle, counting disabled connections too since crossover can enable them again
        /// </summary>
        private void AddConnectionMutation(IInnovationGenerator generator, NeatSettings settings)
        {
            Random random = NEATUtils.Random;
            List<INodeGene> sources = Nodes.Values.Where(CanBeSource).ToList();
            List<INodeGene> targets = Nodes.Values.Where(CanBeTarget).ToList();

            if (sources.Count == 0 || targets.Count == 0) return;

            for (int attempt = 0; attempt < settings.AddConnectionAttempts; attempt++)
            {
                INodeGene fromNode = sources.RandomElement(random);
                INodeGene toNode = targets.RandomElement(random);

                if (fromNode.Innovation == toNode.Innovation) continue;

                bool hasConnection = Connections.Values.Any(x => x.Input.Innovation == fromNode.Innovation && x.Output.Innovation == toNode.Innovation);
                if (hasConnection || IsReachable(toNode.Innovation, fromNode.Innovation)) continue;

                uint innovation = generator.GetConnectionInnovationNumber(fromNode.Innovation, toNode.Innovation);
                Connections.Add(innovation, new ConnectionGene(innovation, fromNode, toNode, RandomWeight(settings)));
                return;
            }
        }

        /// <summary>
        /// Whether a path leads from one node to the other through enabled or disabled connections
        /// </summary>
        private bool IsReachable(uint from, uint to)
        {
            ILookup<uint, uint> successors = Connections.Values.ToLookup(x => x.Input.Innovation, x => x.Output.Innovation);
            var visited = new HashSet<uint> { from };
            var stack = new Stack<uint>();
            stack.Push(from);

            while (stack.Count > 0)
            {
                uint node = stack.Pop();
                if (node == to) return true;

                foreach (uint next in successors[node].Where(x => !visited.Contains(x)))
                {
                    visited.Add(next);
                    stack.Push(next);
                }
            }

            return false;
        }

        /// <summary>
        /// Splits an enabled connection: the old connection is disabled, the connection into the new node gets
        /// weight 1 and the connection out of it the old weight, so the network behaves almost as before
        /// </summary>
        private void AddNodeMutation(IInnovationGenerator generator)
        {
            List<IConnectionGene> enabled = Connections.Values.Where(x => x.Enabled).ToList();
            if (enabled.Count == 0) return;

            IConnectionGene connection = enabled.RandomElement(NEATUtils.Random);

            uint nodeId = generator.GetNodeInnovationNumber(connection.Innovation);
            if (Nodes.ContainsKey(nodeId))
            {
                // this genome split the same connection before (it was enabled again by crossover)
                nodeId = generator.NewNodeInnovationNumber();
            }

            connection.Disable();
            INodeGene newNode = AddNode(nodeId, NodeType.Hidden);

            uint innovation = generator.GetConnectionInnovationNumber(connection.Input.Innovation, newNode.Innovation);
            Connections.Add(innovation, new ConnectionGene(innovation, connection.Input, newNode, 1f));

            innovation = generator.GetConnectionInnovationNumber(newNode.Innovation, connection.Output.Innovation);
            Connections.Add(innovation, new ConnectionGene(innovation, newNode, connection.Output, connection.Weight));
        }

        private void OnFitnessAssinged()
        {
            fitnessAssigned?.Invoke(this, new FitnessAssignedEventArgs(this));
        }

        private void MutateAllConnections(NeatSettings settings)
        {
            Random random = NEATUtils.Random;
            foreach (var connection in Connections.Values)
            {
                if (random.NextDouble() < settings.WeightPerturbationChance)
                {
                    float weight = connection.Weight + random.NextFloat(-settings.WeightPerturbation, settings.WeightPerturbation);
                    connection.Weight = Math.Clamp(weight, -settings.MaxWeight, settings.MaxWeight);
                }
                else
                {
                    connection.Weight = RandomWeight(settings);
                }
            }
        }
    }
}
