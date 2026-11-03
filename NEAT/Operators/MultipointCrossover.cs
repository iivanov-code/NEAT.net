using System;
using NEAT.Net.Abstrations;
using NEAT.Net.Abstrations.Enums;
using NEAT.Net.Abstrations.Interfaces;

namespace NEAT.Net.Operators
{
    /// <summary>
    /// The crossover of the NEAT paper: matching genes are inherited randomly from either parent, disjoint and excess
    /// genes from the fitter parent. A gene disabled in either parent is disabled with
    /// <see cref="NeatSettings.DisableInheritedChance"/>. The child has the topology of the fitter parent, so it
    /// stays acyclic
    /// </summary>
    public class MultipointCrossover : ICrossover
    {
        public IGenome Cross(IGenome fitter, IGenome weaker, EvolutionContext context)
        {
            ArgumentNullException.ThrowIfNull(fitter);
            ArgumentNullException.ThrowIfNull(weaker);
            ArgumentNullException.ThrowIfNull(context);

            Random random = context.Random;
            Genome child = new Genome();

            foreach (INodeGene node in fitter.Nodes.Values)
            {
                child.Nodes.Add(node.Innovation, InheritNode(node, weaker.Nodes.GetValueOrDefault(node.Innovation), context));
            }

            foreach (var gene in GenomeOperations.AlignGenes(fitter.Connections, weaker.Connections))
            {
                if (gene.First == null)
                {
                    continue; // disjoint and excess genes of the weaker parent are not inherited
                }

                IConnectionGene connection;
                if (gene.GeneType == GeneType.Matching)
                {
                    connection = InheritMatching(child, gene.First, gene.Second, context);

                    if (!gene.First.Enabled || !gene.Second.Enabled)
                    {
                        if (random.NextDouble() < context.Settings.DisableInheritedChance)
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
                    connection = GenomeOperations.CopyConnection(child, gene.First);
                }

                child.Connections.Add(connection.Innovation, connection);
            }

            return child;
        }

        /// <summary>
        /// Gene of a matching connection: a copy of either parent's gene, chosen at random
        /// </summary>
        protected virtual IConnectionGene InheritMatching(IGenome child, IConnectionGene fitter, IConnectionGene weaker, EvolutionContext context)
        {
            IConnectionGene source = context.Random.GetRandomBool() ? fitter : weaker;
            return GenomeOperations.CopyConnection(child, source);
        }

        /// <summary>
        /// Node gene of the child; <paramref name="weaker"/> is the same node of the other parent, or null
        /// </summary>
        protected virtual INodeGene InheritNode(INodeGene fitter, INodeGene weaker, EvolutionContext context)
        {
            return fitter.Clone();
        }
    }
}
