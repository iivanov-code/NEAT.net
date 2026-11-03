using System;
using System.Collections.Generic;
using System.Linq;
using NEAT.Net.Abstrations;
using NEAT.Net.Abstrations.Enums;
using NEAT.Net.Abstrations.Interfaces;
using NEAT.Net.Operators;
using ActivationTypes = ANN.Net.Abstractions.Enums.ActivationTypes;

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
        /// Minimal starting genome: every input (and the bias) connected directly to every output, without hidden nodes.
        /// Outputs get an activation of <see cref="NeatSettings.OutputActivations"/>
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
                outputs.Add(genome.AddNode(generator.NewNodeInnovationNumber(), NodeType.Output, GenomeOperations.PickActivation(settings.OutputActivations, NEATUtils.Random)));
            }

            foreach (INodeGene from in sources)
            {
                foreach (INodeGene to in outputs)
                {
                    uint innovation = generator.GetConnectionInnovationNumber(from.Innovation, to.Innovation);
                    genome.Connections.Add(innovation, new ConnectionGene(innovation, from, to, GenomeOperations.RandomWeight(settings, NEATUtils.Random)));
                }
            }

            return genome;
        }

        /// <summary>
        /// Used by the overloads without settings
        /// </summary>
        private static readonly NeatSettings DefaultSettings = new NeatSettings();

        // the operators behind Mutate and Crossover
        private static readonly IMutation Weights = new WeightMutation();
        private static readonly IMutation AddConnection = new AddConnectionMutation();
        private static readonly IMutation SplitConnection = new AddNodeMutation();
        private static readonly IMutation ChangeActivation = new ActivationMutation();
        private static readonly IMutation ToggleEnable = new ToggleEnableMutation();
        private static readonly IMutation RemoveConnection = new RemoveConnectionMutation();
        private static readonly IMutation RemoveNode = new RemoveNodeMutation();
        private static readonly ICrossover DefaultCrossover = new MultipointCrossover();

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
                genome.Connections.Add(connection.Innovation, GenomeOperations.CopyConnection(genome, connection));
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

            foreach (var gene in GenomeOperations.AlignGenes(Connections, other.Connections))
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

            var context = new EvolutionContext(null, settings, NEATUtils.Random);
            var (fitter, weaker) = GenomeOperations.OrderParents(this, other, context.Random);
            return DefaultCrossover.Cross(fitter, weaker, context);
        }

        public void Mutate(MutationType mutationType, IInnovationGenerator generator)
        {
            Mutate(mutationType, generator, DefaultSettings);
        }

        public void Mutate(MutationType mutationType, IInnovationGenerator generator, NeatSettings settings)
        {
            ArgumentNullException.ThrowIfNull(settings);

            IMutation mutation = mutationType switch
            {
                MutationType.Connection => AddConnection,
                MutationType.Node => SplitConnection,
                MutationType.AllConnections => Weights,
                MutationType.Activation => ChangeActivation,
                MutationType.ToggleEnable => ToggleEnable,
                MutationType.RemoveConnection => RemoveConnection,
                MutationType.RemoveNode => RemoveNode,
                _ => throw new ArgumentException("Unknown mutation type", nameof(mutationType))
            };

            mutation.Apply(this, new EvolutionContext(generator, settings, NEATUtils.Random));
        }

        private INodeGene AddNode(uint innovation, NodeType type, ActivationTypes? activation = null)
        {
            INodeGene node = new NodeGene(innovation, type, activation);
            Nodes.Add(innovation, node);
            return node;
        }

        private void OnFitnessAssinged()
        {
            fitnessAssigned?.Invoke(this, new FitnessAssignedEventArgs(this));
        }
    }
}
