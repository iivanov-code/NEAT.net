using System;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Spiking;
using ANN.Net.Spiking;

namespace NEAT.Net.Networks
{
    /// <summary>
    /// How a genome is turned into a spiking network: every hidden and output node becomes a spiking neuron whose
    /// spike function follows its activation gene, and the outputs are the firing rates of the output nodes
    /// </summary>
    public class SpikingPhenotypeSettings
    {
        /// <summary>
        /// Time steps every input is presented for; the outputs are firing rates in steps of 1 / TimeSteps
        /// </summary>
        public int TimeSteps { get; set; } = 20;

        /// <summary>
        /// Normalization of the raw inputs; null uses them as they are, so they should already lie in [0, 1]
        /// </summary>
        public InputScaling InputScaling { get; set; }

        /// <summary>
        /// Creates the spike function of the inputs for every network; null uses an integrate-and-fire neuron with
        /// threshold 1, so an input of 0.25 fires every fourth step
        /// </summary>
        public Func<ISpikingActivation> InputActivation { get; set; }

        /// <summary>
        /// Creates the spike function of the hidden and output nodes with an activation gene, once per network and
        /// activation (every network gets its own, so random spike functions are not shared between threads)
        /// </summary>
        public Func<ActivationTypes, ISpikingActivation> Activation { get; set; } = DefaultActivation;

        /// <summary>
        /// Activation of hidden nodes whose gene has none
        /// </summary>
        public ActivationTypes HiddenActivationType { get; set; } = ActivationTypes.Sigmoid;

        /// <summary>
        /// Activation of output nodes whose gene has none
        /// </summary>
        public ActivationTypes OutputActivationType { get; set; } = ActivationTypes.Sigmoid;

        /// <summary>
        /// Whether <see cref="ISpikingNetwork.Learn"/> changes the synapses (spike-timing-dependent plasticity)
        /// </summary>
        public bool Learn { get; set; }

        public StdpOptions Plasticity { get; set; }

        /// <summary>
        /// The spike function whose firing rate follows the activation: integrate-and-fire for ReLU, a neuron that
        /// also fires negative spikes for leaky ReLU and the identity, the sigmoid rate neuron for sigmoid and tanh,
        /// and integrate-and-fire for every other activation
        /// </summary>
        public static ISpikingActivation DefaultActivation(ActivationTypes activation)
        {
            switch (activation)
            {
                case ActivationTypes.LeakyRELU:
                    return new SignedIntegrateAndFireActivation(negativeSlope: 0.01f);
                case ActivationTypes.Identity:
                    return new SignedIntegrateAndFireActivation(negativeSlope: 1);
                case ActivationTypes.Sigmoid:
                    return new SigmoidRateActivation();
                case ActivationTypes.HyperbolicTangens:
                    return new SigmoidRateActivation(gain: 2);
                default:
                    return new IntegrateAndFireActivation();
            }
        }
    }
}
