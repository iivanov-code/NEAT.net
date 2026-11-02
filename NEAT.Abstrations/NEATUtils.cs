using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;

namespace NEAT.Net.Abstrations
{
    public static class NEATUtils
    {
        private static int id = -1;

        /// <summary>
        /// Random source used by the algorithm. Defaults to the thread-safe <see cref="System.Random.Shared"/>;
        /// assign a seeded instance for reproducible runs (it must then be used from a single thread)
        /// </summary>
        public static Random Random { get; set; } = Random.Shared;

        public static uint NewID()
        {
            return (uint)Interlocked.Increment(ref id);
        }

        public static TValue GetValueOrDefault<TKey, TValue>(this IDictionary<TKey, TValue> dictionary, TKey key)
        {
            return dictionary.TryGetValue(key, out TValue value) ? value : default;
        }

        public static void Swap<T>(ref T first, ref T second)
        {
            T temp = first;
            first = second;
            second = temp;
        }

        /// <summary>
        /// Uniform random number in [minimum, maximum)
        /// </summary>
        public static float NextFloat(this Random rand, float minimum, float maximum)
        {
            return (float)(rand.NextDouble() * (maximum - minimum) + minimum);
        }

        public static float GetRandomNumber(this Random rand, float minimum, float maximum, int decimals = 2)
        {
            return (float)Math.Round(rand.NextFloat(minimum, maximum), decimals, MidpointRounding.AwayFromZero);
        }

        public static bool GetRandomBool(this Random rand)
        {
            return rand.Next(2) == 0;
        }

        public static TValue GetElementAt<TKey, TValue>(this IDictionary<TKey, TValue> dictionary, int index)
        {
            return dictionary.Values.ElementAt(index);
        }

        public static T RandomElement<T>(this IReadOnlyList<T> list, Random rand)
        {
            return list[rand.Next(list.Count)];
        }
    }
}
