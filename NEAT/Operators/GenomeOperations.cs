using System;
using System.Collections.Generic;
using System.Linq;
using NEAT.Net.Abstrations;
using NEAT.Net.Abstrations.Enums;
using NEAT.Net.Abstrations.Interfaces;
using NEAT.Net.Models;
using ActivationTypes = ANN.Net.Abstractions.Enums.ActivationTypes;

namespace NEAT.Net.Operators
{
    /// <summary>
    /// Building blocks shared by the genome and the mutation and crossover operators
    /// </summary>
    internal static class GenomeOperations
    {
        /// <summary>
        /// Lines up the genes of two genomes by innovation number. Genes of one genome that fall inside the innovation
        /// range of the other are disjoint, genes beyond the other genome's highest innovation are excess
        /// </summary>
        public static IEnumerable<GeneCompareType<TValue>> AlignGenes<TValue>(IDictionary<uint, TValue> first, IDictionary<uint, TValue> second)
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

        /// <summary>
        /// The fitter parent first; equally fit parents in random order
        /// </summary>
        public static (IGenome Fitter, IGenome Weaker) OrderParents(IGenome first, IGenome second, Random random)
        {
            int comparison = first.Fitness.CompareTo(second.Fitness);
            return comparison > 0 || (comparison == 0 && random.GetRandomBool()) ? (first, second) : (second, first);
        }

        /// <summary>
        /// Copies a connection gene of any genome and points it to the nodes of <paramref name="target"/>
        /// </summary>
        public static IConnectionGene CopyConnection(IGenome target, IConnectionGene connection)
        {
            IConnectionGene copy = connection.Clone();
            copy.Input = target.Nodes[connection.Input.Innovation];
            copy.Output = target.Nodes[connection.Output.Innovation];
            return copy;
        }

        /// <summary>
        /// Whether a path leads from one node to the other through enabled or disabled connections
        /// </summary>
        public static bool IsReachable(IGenome genome, uint from, uint to)
        {
            ILookup<uint, uint> successors = genome.Connections.Values.ToLookup(x => x.Input.Innovation, x => x.Output.Innovation);
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

        public static float RandomWeight(NeatSettings settings, Random random)
        {
            return random.NextFloat(-settings.InitialWeightRange, settings.InitialWeightRange);
        }

        /// <summary>
        /// Random activation of the list; a list with one entry draws no random number, so runs with a single
        /// activation follow the same random sequence as before activations were part of genes
        /// </summary>
        public static ActivationTypes PickActivation(IList<ActivationTypes> activations, Random random)
        {
            return activations.Count == 1 ? activations[0] : activations[random.Next(activations.Count)];
        }

        public static bool CanBeSource(INodeGene node) => node.Type != NodeType.Output;

        public static bool CanBeTarget(INodeGene node) => node.Type == NodeType.Hidden || node.Type == NodeType.Output;
    }
}
