using System;
using System.Collections.Generic;
using ANN.Net.Abstractions.Enums;
using NEAT.Net.Abstrations;

namespace NEAT.Net.Benchmarks
{
    /// <summary>
    /// Settings compared by the benchmarks
    /// </summary>
    internal static class Configurations
    {
        /// <summary>
        /// sigmoid: the defaults; mixed: five hidden activations; mixed-c4: the same with the activation term in speciation
        /// </summary>
        public static NeatSettings Create(string name)
        {
            var mixed = new List<ActivationTypes>
            {
                ActivationTypes.Sigmoid, ActivationTypes.HyperbolicTangens, ActivationTypes.Gaussian, ActivationTypes.Sinusoid, ActivationTypes.RELU
            };

            return name switch
            {
                "sigmoid" => new NeatSettings(),
                "mixed" => new NeatSettings { HiddenActivations = mixed },
                "mixed-c4" => new NeatSettings { HiddenActivations = mixed, ActivationDifferenceCoefficient = 1 },
                _ => throw new ArgumentException($"Unknown configuration {name}", nameof(name))
            };
        }
    }
}
