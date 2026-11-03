using System;
using System.Linq;
using ANN.Net.Abstractions.Interfaces;
using NEAT.Net.Abstrations;
using NEAT.Net.Abstrations.Interfaces;
using NEAT.Net.Networks;

namespace NEAT.Net.Benchmarks
{
    /// <summary>
    /// Fitness (n - Σ|error|)², n the number of cases, as in the XOR experiment of the NEAT paper
    /// </summary>
    internal sealed class ProblemEvaluator : NetworkEvaluator
    {
        private readonly Problem problem;

        public ProblemEvaluator(Problem problem, IGenome genome, IInnovationGenerator generator, NeatSettings settings)
            : base(genome, generator, settings)
        {
            this.problem = problem;
        }

        public float[] Outputs(IGraphNetwork network) => problem.Inputs.Select(x => (float)network.Propagate(x.Select(v => (Quad)v).ToArray())[0]).ToArray();

        protected override float EvaluateNetwork(IGraphNetwork network, IGenome genome)
        {
            float[] outputs = Outputs(network);
            float error = outputs.Zip(problem.Targets, (o, t) => Math.Abs(o - t)).Sum();
            float score = Math.Max(0, problem.Targets.Length - error);
            return score * score;
        }
    }
}
