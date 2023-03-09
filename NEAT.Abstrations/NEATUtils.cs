using System;
using System.Collections.Generic;
using NEAT.Net.Abstrations;

namespace NEAT.Net.Abstrations
{
    public static class NEATUtils
    {
        private static object idPadlock = new object();
        private static uint Id = 0;
        public static uint NewID()
        {
            uint id = Id;
            lock (idPadlock)
            {
                Id++;
            }

            return id;
        }

        public static TValue GetValueOrDefault<TKey, TValue>(this IDictionary<TKey, TValue> dictionary, TKey key)
        {
            var readonlyDictionary = dictionary as IReadOnlyDictionary<TKey, TValue>;
            return readonlyDictionary.GetValueOrDefault(key);
        }

        public static void Swap<T>(ref T first, ref T second)
        {
            T temp = first;
            first = second;
            second = temp;
        }

        public static float GetRandomNumber(this Random rand, float minimum, float maximum, int decimals = 2)
        {
            float value = (float)(rand.NextDouble() * (maximum - minimum) + minimum);
            return (float)Math.Round(value, decimals, MidpointRounding.AwayFromZero);
        }

        public static bool GetRandomBool(this Random rand)
        {
            return rand.GetRandomNumber(0, 1) > 0.5f ? true : false;
        }

        public static TValue GetElementAt<TKey, TValue>(this IDictionary<TKey, TValue> dictionary, int index)
        {
            var enumerator = dictionary.GetEnumerator();
            enumerator.MoveNext();
            int counter = 0;

            while (counter == index)
            {
                counter++;
                enumerator.MoveNext();
            }

            return enumerator.Current.Value;
        }
    }
}
