using System;
using System.Collections.Generic;
using System.Linq;

namespace NEAT.Net.Benchmarks
{
    /// <summary>
    /// Supervised task: inputs, expected outputs in [0, 1] and when a network counts as a solution
    /// </summary>
    internal sealed class Problem
    {
        private Problem(string name, float[][] inputs, float[] targets, Func<float[], bool> isSolved)
        {
            Name = name;
            Inputs = inputs;
            Targets = targets;
            IsSolved = isSolved;
        }

        public string Name { get; }

        public float[][] Inputs { get; }

        public float[] Targets { get; }

        /// <summary>
        /// Takes the outputs for all inputs
        /// </summary>
        public Func<float[], bool> IsSolved { get; }

        public static IReadOnlyList<Problem> All { get; } = new[] { Xor(), Circle(), Sine() };

        /// <summary>
        /// Every case classified on the right side of 0.5
        /// </summary>
        private static Problem Xor()
        {
            float[][] inputs = { new[] { 0f, 0f }, new[] { 0f, 1f }, new[] { 1f, 0f }, new[] { 1f, 1f } };
            float[] targets = { 0, 1, 1, 0 };
            return new Problem("xor", inputs, targets, outputs => Classified(outputs, targets) == targets.Length);
        }

        /// <summary>
        /// Inside or outside a circle of radius 0.6 around the origin, on a 7 x 7 grid in [-1, 1]²; solved at 95% accuracy
        /// </summary>
        private static Problem Circle()
        {
            var inputs = new List<float[]>();
            var targets = new List<float>();
            for (int i = 0; i < 7; i++)
            {
                for (int j = 0; j < 7; j++)
                {
                    float x = -1 + (i / 3f), y = -1 + (j / 3f);
                    inputs.Add(new[] { x, y });
                    targets.Add((x * x) + (y * y) < 0.36f ? 1 : 0);
                }
            }

            float[] expected = targets.ToArray();
            return new Problem("circle", inputs.ToArray(), expected, outputs => Classified(outputs, expected) >= 0.95 * expected.Length);
        }

        /// <summary>
        /// (sin(x) + 1) / 2 for 25 points of [-π, π] (input x / π); solved below a mean absolute error of 0.05
        /// </summary>
        private static Problem Sine()
        {
            float[][] inputs = Enumerable.Range(0, 25).Select(i => new[] { -1 + (i / 12f) }).ToArray();
            float[] targets = inputs.Select(x => (float)((Math.Sin(x[0] * Math.PI) + 1) / 2)).ToArray();
            return new Problem("sine", inputs, targets, outputs => outputs.Zip(targets, (o, t) => Math.Abs(o - t)).Average() < 0.05);
        }

        private static int Classified(float[] outputs, float[] targets) => outputs.Zip(targets).Count(x => Math.Abs(x.First - x.Second) < 0.5f);
    }
}
