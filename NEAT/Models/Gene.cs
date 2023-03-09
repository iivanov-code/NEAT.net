using NEAT.Net.Abstrations.Enums;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NEAT.Net.Models
{
    internal class GeneCompareType<T>
    {
        public bool Fittest { get; set; }
        public T Gene { get; set; }
        public T MatchingGene { get; set; }
        public GeneType GeneType { get; set; }
    }
}
