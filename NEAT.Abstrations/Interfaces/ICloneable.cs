namespace NEAT.Net.Abstrations.Interfaces
{
    public interface ICloneable<out T>
    {
        T Clone();
    }
}
