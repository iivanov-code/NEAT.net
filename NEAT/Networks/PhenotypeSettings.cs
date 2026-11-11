using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Settings;

namespace NEAT.Net.Networks
{
    /// <summary>
    /// How a genome is turned into an ANN.Net network
    /// </summary>
    public class PhenotypeSettings
    {
        public ActivationTypes InputActivationType { get; set; } = ActivationTypes.Identity;

        /// <summary>
        /// Activation of hidden nodes whose gene has none (genomes saved before activations were part of genes).
        /// New hidden nodes get theirs from <c>NeatSettings.HiddenActivations</c>
        /// </summary>
        public ActivationTypes HiddenActivationType { get; set; } = ActivationTypes.Sigmoid;

        /// <summary>
        /// Activation of output nodes whose gene has none (genomes saved before activations were part of genes).
        /// New genomes get theirs from <c>NeatSettings.OutputActivations</c>
        /// </summary>
        public ActivationTypes OutputActivationType { get; set; } = ActivationTypes.Sigmoid;

        /// <summary>
        /// Optimizer of every synapse, used only when the network is also trained with backpropagation
        /// </summary>
        public OptimizerTypes OptimizerType { get; set; } = OptimizerTypes.Normal;

        /// <summary>
        /// Loss the network trains on when it is also trained with backpropagation
        /// </summary>
        public LossFunctionTypes LossType { get; set; } = LossFunctionTypes.Squared;

        /// <summary>
        /// Parameters of the loss, such as δ of the Huber loss; null for the defaults
        /// </summary>
        public LossOptions LossOptions { get; set; }

        public bool OutputApplySoftMax { get; set; }

        public IUniqueIdentityGenerator IdentityGenerator { get; set; }
    }
}
