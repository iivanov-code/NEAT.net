using System;
using NEAT.Net.Abstrations;
using NEAT.Net.Abstrations.Enums;
using NEAT.Net.Abstrations.Interfaces;

namespace NEAT.Net.Operators
{
    /// <summary>
    /// The crossover of the NEAT paper: matching genes are inherited randomly from either parent, disjoint and excess
    /// genes from the fitter parent. A gene disabled in either parent is disabled with
    /// <see cref="NeatSettings.DisableInheritedChance"/>. Nodes both parents have take their activation from a random
    /// parent (no random number is drawn when the activations are the same). The child has the topology of the
    /// fitter parent, so it stays acyclic
    /// </summary>
    public class MultipointCrossover : ICrossover
    {
        /// <summary>
        /// Chooses the child's gene for a connection both parents have
        /// </summary>
        protected delegate IConnectionGene MatchingGeneSelector(IGenome child, IConnectionGene fitter, IConnectionGene weaker, EvolutionContext context);

        public virtual IGenome Cross(IGenome fitter, IGenome weaker, EvolutionContext context)
        {
            return Combine(fitter, weaker, context, InheritMatching);
        }

        /// <summary>
        /// Builds the child: the nodes and the disjoint and excess genes of <paramref name="fitter"/>, matching genes
        /// chosen by <paramref name="inheritMatching"/>
        /// </summary>
        protected IGenome Combine(IGenome fitter, IGenome weaker, EvolutionContext context, MatchingGeneSelector inheritMatching)
        {
            ArgumentNullException.ThrowIfNull(fitter);
            ArgumentNullException.ThrowIfNull(weaker);
            ArgumentNullException.ThrowIfNull(context);
            ArgumentNullException.ThrowIfNull(inheritMatching);

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
                    connection = inheritMatching(child, gene.First, gene.Second, context);

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
        /// Node gene of the child; <paramref name="weaker"/> is the same node of the other parent, or null. A node both
        /// parents have takes the activation of a random parent
        /// </summary>
        protected virtual INodeGene InheritNode(INodeGene fitter, INodeGene weaker, EvolutionContext context)
        {
            INodeGene node = fitter.Clone();
            if (weaker != null && weaker.Activation != fitter.Activation && context.Random.GetRandomBool())
            {
                node.Activation = weaker.Activation;
            }

            return node;
        }
    }
}
