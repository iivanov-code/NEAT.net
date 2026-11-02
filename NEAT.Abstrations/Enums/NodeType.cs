namespace NEAT.Net.Abstrations.Enums
{
    public enum NodeType
    {
        Input = 0,
        Hidden = 1,
        Output = 2,

        /// <summary>
        /// Input node with the constant value 1; the weights of its connections are the biases of the other nodes
        /// </summary>
        Bias = 3
    }
}
