using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json.Nodes;
using ActivationTypes = ANN.Net.Abstractions.Enums.ActivationTypes;
using NEAT.Net.Abstrations;
using NEAT.Net.Abstrations.Enums;
using NEAT.Net.Abstrations.Interfaces;
using Xunit;

namespace NEAT.Net.Tests
{
    [Collection(RandomCollection.Name)]
    public class NeatSettingsTests
    {
        private sealed class SizeEvaluator : Evaluator
        {
            public SizeEvaluator(IGenome genome, IInnovationGenerator generator, NeatSettings settings)
                : base(genome, generator, settings)
            { }

            public override float EvaluateGenome(IGenome genome) => genome.Connections.Count + genome.Nodes.Count;
        }

        private static SizeEvaluator Create(NeatSettings settings)
        {
            var generator = new InnovationGenerator();
            return new SizeEvaluator(Genome.CreateMinimal(3, 2, generator, settings: settings), generator, settings);
        }

        [Fact]
        public void Defaults_AreTheValuesOfDefaultConstants()
        {
            var settings = new NeatSettings();

            Assert.Equal(DefaultConstants.POPULATION_SIZE, settings.PopulationSize);
            Assert.Equal(DefaultConstants.DT, settings.CompatibilityThreshold);
            Assert.Equal(DefaultConstants.C1, settings.ExcessCoefficient);
            Assert.Equal(DefaultConstants.C2, settings.DisjointCoefficient);
            Assert.Equal(DefaultConstants.C3, settings.WeightDifferenceCoefficient);
            Assert.Equal(0f, settings.ActivationDifferenceCoefficient);
            Assert.Equal(DefaultConstants.SMALL_GENOME_SIZE, settings.SmallGenomeSize);
            Assert.Equal(DefaultConstants.SPECIES_WITH_MORE_THAN, settings.ElitismMinimumSpecieSize);
            Assert.Equal(DefaultConstants.STAGNATION_LIMIT, settings.StagnationLimit);
            Assert.Equal(DefaultConstants.SURVIVAL_THRESHOLD, settings.SurvivalThreshold);
            Assert.Equal(DefaultConstants.PERCENT_OF_DIRECT_OFFSPRING, settings.MutationOnlyRate);
            Assert.Equal(DefaultConstants.INTERSPECIES_MATING_RATE, settings.InterspeciesMatingRate);
            Assert.Equal(DefaultConstants.DISABLED_WEIGHT_CHANCE, settings.DisableInheritedChance);
            Assert.Equal(DefaultConstants.CHANCE_OF_WEIGHTS_MUTATION, settings.WeightMutationRate);
            Assert.Equal(DefaultConstants.PERTURBED_CHANCE, settings.WeightPerturbationChance);
            Assert.Equal(DefaultConstants.WEIGHT_PERTURBATION, settings.WeightPerturbation);
            Assert.Equal(DefaultConstants.INITIAL_WEIGHT_RANGE, settings.InitialWeightRange);
            Assert.Equal(DefaultConstants.MAX_WEIGHT, settings.MaxWeight);
            Assert.Equal(DefaultConstants.PROBABILITY_OF_ADDING_NEW_NODE, settings.AddNodeRate);
            Assert.Equal(DefaultConstants.PROBABILITY_OF_ADDING_NEW_LINK, settings.AddConnectionRate);
            Assert.Equal(DefaultConstants.ADD_LINK_ATTEMPTS, settings.AddConnectionAttempts);
            Assert.Equal(new[] { ActivationTypes.Sigmoid }, settings.HiddenActivations);
            Assert.Equal(new[] { ActivationTypes.Sigmoid }, settings.OutputActivations);
            settings.Validate();
        }

        [Fact]
        public void Clone_IsIndependent()
        {
            var settings = new NeatSettings { PopulationSize = 10 };
            NeatSettings clone = settings.Clone();

            clone.PopulationSize = 20;
            clone.HiddenActivations.Add(ActivationTypes.Gaussian);

            Assert.Equal(10, settings.PopulationSize);
            Assert.Single(settings.HiddenActivations);
        }

        [Theory]
        [InlineData(nameof(NeatSettings.PopulationSize))]
        [InlineData(nameof(NeatSettings.CompatibilityThreshold))]
        [InlineData(nameof(NeatSettings.SurvivalThreshold))]
        [InlineData(nameof(NeatSettings.AddNodeRate))]
        [InlineData(nameof(NeatSettings.WeightMutationRate))]
        [InlineData(nameof(NeatSettings.MaxWeight))]
        [InlineData(nameof(NeatSettings.StagnationLimit))]
        [InlineData(nameof(NeatSettings.HiddenActivations))]
        [InlineData(nameof(NeatSettings.OutputActivations))]
        public void Validate_RejectsInvalidValues(string setting)
        {
            var settings = new NeatSettings();
            switch (setting)
            {
                case nameof(NeatSettings.PopulationSize): settings.PopulationSize = 0; break;
                case nameof(NeatSettings.CompatibilityThreshold): settings.CompatibilityThreshold = float.NaN; break;
                case nameof(NeatSettings.SurvivalThreshold): settings.SurvivalThreshold = 0; break;
                case nameof(NeatSettings.AddNodeRate): settings.AddNodeRate = 1.5f; break;
                case nameof(NeatSettings.WeightMutationRate): settings.WeightMutationRate = -0.1f; break;
                case nameof(NeatSettings.MaxWeight): settings.MaxWeight = 0; break;
                case nameof(NeatSettings.StagnationLimit): settings.StagnationLimit = 0; break;
                case nameof(NeatSettings.HiddenActivations): settings.HiddenActivations.Clear(); break;
                case nameof(NeatSettings.OutputActivations): settings.OutputActivations = new[] { ActivationTypes.BReLU }; break;
            }

            ArgumentException exception = Assert.ThrowsAny<ArgumentException>(settings.Validate);
            Assert.Equal(setting, exception.ParamName);
        }

        [Fact]
        public void Evaluator_RejectsInvalidSettings_AtStartAndBeforeEveryGeneration()
        {
            Assert.ThrowsAny<ArgumentException>(() => Create(new NeatSettings { MutationOnlyRate = 2 }));

            SizeEvaluator evaluator = Create(new NeatSettings { PopulationSize = 20 });
            evaluator.Settings.AddConnectionRate = -1;
            Assert.ThrowsAny<ArgumentException>(evaluator.Evaluate);
        }

        [Fact]
        public void Settings_ChangedDuringARun_TakeEffectInTheNextGeneration()
        {
            using var seed = new SeededRandom(21);
            SizeEvaluator evaluator = Create(new NeatSettings { PopulationSize = 30, AddNodeRate = 0, AddConnectionRate = 0 });

            evaluator.Evaluate();
            Assert.Equal(30, evaluator.Genomes.Count);
            Assert.All(evaluator.Genomes, x => Assert.DoesNotContain(x.Nodes.Values, n => n.Type == NodeType.Hidden));

            evaluator.Settings.PopulationSize = 50;
            evaluator.Settings.AddNodeRate = 1;
            evaluator.Settings.ElitismMinimumSpecieSize = int.MaxValue; // no unchanged champions

            evaluator.Evaluate();
            Assert.Equal(50, evaluator.Genomes.Count);
            Assert.All(evaluator.Genomes, x => Assert.Contains(x.Nodes.Values, n => n.Type == NodeType.Hidden));
        }

        [Fact]
        public void WeightSettings_LimitTheWeights()
        {
            using var seed = new SeededRandom(22);
            var settings = new NeatSettings { InitialWeightRange = 0.1f, MaxWeight = 0.3f, WeightPerturbationChance = 1 };
            var generator = new InnovationGenerator();
            Genome genome = Genome.CreateMinimal(4, 3, generator, settings: settings);

            Assert.All(genome.Connections.Values, x => Assert.InRange(x.Weight, -0.1f, 0.1f));

            for (int i = 0; i < 50; i++)
            {
                genome.Mutate(MutationType.AllConnections, generator, settings);
            }

            Assert.All(genome.Connections.Values, x => Assert.InRange(x.Weight, -0.3f, 0.3f));
            Assert.Contains(genome.Connections.Values, x => Math.Abs(x.Weight) >= 0.3f); // clamped at the limit
        }

        [Fact]
        public void CompatibilityDistance_UsesTheCoefficientsOfTheSettings()
        {
            using var seed = new SeededRandom(23);
            var generator = new InnovationGenerator();
            Genome first = Genome.CreateMinimal(3, 2, generator);
            IGenome second = first.Clone();
            second.Mutate(MutationType.Node, generator);
            second.Mutate(MutationType.AllConnections, generator);

            var settings = new NeatSettings { ExcessCoefficient = 2, DisjointCoefficient = 3, WeightDifferenceCoefficient = 0.7f };
            Assert.Equal(first.CompatibilityDistance(second, 2, 3, 0.7f), first.CompatibilityDistance(second, settings));

            // with N taken from the genome size instead of 1, the gene counts are divided by 10 connections
            settings.SmallGenomeSize = 0;
            settings.WeightDifferenceCoefficient = 0;
            Assert.Equal(first.CompatibilityDistance(second, 2, 3, 0) / 10, first.CompatibilityDistance(second, settings), 5);
        }

        [Fact]
        public void Checkpoint_RestoresTheSettings()
        {
            SizeEvaluator evaluator = Create(new NeatSettings
            {
                PopulationSize = 25,
                CompatibilityThreshold = 1.75f,
                AddNodeRate = 0.2f,
                HiddenActivations = new[] { ActivationTypes.Sigmoid, ActivationTypes.Gaussian, ActivationTypes.Sinusoid }
            });
            evaluator.Evaluate();

            using var stream = new MemoryStream();
            evaluator.SaveCheckpoint(stream);
            stream.Position = 0;
            SizeEvaluator resumed = Create(new NeatSettings());
            resumed.LoadCheckpoint(stream);

            Assert.Equal(25, resumed.PopulationSize);
            Assert.Equal(1.75f, resumed.CompatibilityThreshold);
            Assert.Equal(0.2f, resumed.Settings.AddNodeRate);
            Assert.Equal(new[] { ActivationTypes.Sigmoid, ActivationTypes.Gaussian, ActivationTypes.Sinusoid }, resumed.Settings.HiddenActivations);
        }

        [Fact]
        public void Checkpoint_WithoutSettings_KeepsTheEvaluatorsSettingsAndTheSavedThreshold()
        {
            SizeEvaluator evaluator = Create(new NeatSettings { PopulationSize = 25, CompatibilityThreshold = 1.75f });
            evaluator.Evaluate();

            using var stream = new MemoryStream();
            evaluator.SaveCheckpoint(stream);

            // a checkpoint as saved before settings were part of it
            JsonNode json = JsonNode.Parse(stream.ToArray());
            json.AsObject().Remove("settings");
            SizeEvaluator resumed = Create(new NeatSettings { PopulationSize = 40 });
            resumed.LoadCheckpoint(new MemoryStream(Encoding.UTF8.GetBytes(json.ToJsonString())));

            Assert.Equal(40, resumed.PopulationSize);
            Assert.Equal(1.75f, resumed.CompatibilityThreshold);
            Assert.Equal(25, resumed.Genomes.Count);
        }

        [Fact]
        public void Checkpoint_WithInvalidSettings_IsRejected()
        {
            SizeEvaluator evaluator = Create(new NeatSettings { PopulationSize = 10 });
            using var stream = new MemoryStream();
            evaluator.SaveCheckpoint(stream);

            JsonNode json = JsonNode.Parse(stream.ToArray());
            json["settings"]["survivalThreshold"] = 5;

            Assert.Throws<InvalidDataException>(() => Create(new NeatSettings()).LoadCheckpoint(new MemoryStream(Encoding.UTF8.GetBytes(json.ToJsonString()))));
        }
    }
}
