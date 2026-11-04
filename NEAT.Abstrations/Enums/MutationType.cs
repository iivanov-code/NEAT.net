namespace NEAT.Net.Abstrations.Enums
{
    public enum MutationType
    {
        AllConnections,
        Connection,
        Node,

        /// <summary>
        /// Another activation for one hidden (or output) node
        /// </summary>
        Activation,

        /// <summary>
        /// Enables a disabled connection again, or disables an enabled one
        /// </summary>
        ToggleEnable,

        RemoveConnection,

        RemoveNode,
    }
}
