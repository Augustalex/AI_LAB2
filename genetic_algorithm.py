import operator
import random

import numpy as np


class GA(object):
    def __init__(self, populationSize, numberOfGenes, crossoverProbability, mutationProbability,
                 selectionMethod, tournamentSelectionParameter, tournamentSize, numberOfVariables,
                 variableRange, numberOfGenerations, useElitism, numberOfBestIndividualCopies, fitnessFunction):

        self.populationSize = int(populationSize)
        self.numberOfGenes = int(numberOfGenes)
        self.crossoverProbability = float(crossoverProbability)
        self.mutationProbability = float(mutationProbability)
        self.selectionMethod = int(selectionMethod)
        self.tournamentSelectionParameter = float(tournamentSelectionParameter)
        self.tournamentSize = int(tournamentSize)
        self.numberOfVariables = int(numberOfVariables)
        self.variableRange = variableRange
        self.numberOfGenerations = int(numberOfGenerations)
        self.useElitism = bool(useElitism)
        self.numberOfBestIndividualCopies = int(numberOfBestIndividualCopies)
        self.fitness = np.zeros((populationSize, 1))
        self.normalizedFitness = np.zeros((populationSize, 1))
        self.population = self.InitializePopulation(populationSize, numberOfGenes)
        self.vars = np.zeros((populationSize, numberOfVariables))
        self.data = np.zeros((populationSize, numberOfVariables + 1))
        self.generation = 0
        self.EvaluateIndividual = fitnessFunction
        self.CalculateFitness()

    def CalculateFitness(self):
        self.maximumFitness = float("-inf")
        self.minimumFitness = float("inf")
        self.totalFitness = 0.0
        self.averageFitness = 0.0
        self.totalNormalizedFitness = 0.0
        self.averageNormalizedFitness = 0.0

        for i in range(self.populationSize):
            chromosome = self.population[i]
            self.vars[i] = self.DecodeChromosome(chromosome, self.numberOfVariables, self.variableRange)
            self.fitness[i][0] = self.EvaluateIndividual(self.vars[i])
            self.totalFitness = self.totalFitness + self.fitness[i][0]
            if self.fitness[i][0] > self.maximumFitness:
                self.maximumFitness = self.fitness[i][0]
                self.bestIndividualIndex = i
            if self.fitness[i][0] < self.minimumFitness:
                self.minimumFitness = self.fitness[i][0]
            self.data[i][0:self.numberOfVariables] = self.vars[i]
            self.data[i][self.numberOfVariables] = self.fitness[i][0]
        self.averageFitness = self.totalFitness / float(self.populationSize)

        fitnessRange = self.maximumFitness - self.minimumFitness
        if (fitnessRange == 0): fitnessRange = 0.01
        for i in range(self.populationSize):
            self.normalizedFitness[i][0] = (self.fitness[i][0] - self.minimumFitness) / float(fitnessRange)
            self.totalNormalizedFitness = self.totalNormalizedFitness + self.normalizedFitness[i][0]
        self.averageNormalizedFitness = self.totalNormalizedFitness / float(self.populationSize)

    def InitializePopulation(self, populationSize, numberOfGenes):
        population = np.random.random_integers(0, 1, (populationSize, numberOfGenes))
        return population

    def DecodeChromosome(chromosome, nVariables, variableRange):
        nGenes = np.size(chromosome, 0)  # Number of genes in the chromosome
        nBits = nGenes / nVariables  # Number of bits (genes) per variable

        vars = np.zeros(nVariables)  # Create a one-dimensional Numpy array of variables

        # Calculate the value of each variable from the bits in the bit string

        ##############################
        ### YOU'RE CODE GOES HERE ####
        ##############################

        for i in range(0, nVariables):
            value = 0

            power = -1
            for v in range(nBits * i, nBits * i + nBits):
                value += pow(2, power) * chromosome[v]
                power -= 1

            l = variableRange[i][0]
            u = variableRange[i][1]
            vars[i] = l + (u - l) / (1 - pow(2, -nBits)) + value

        return vars

    def RouletteWheelSelect(self, normalizedFitness):
        # Use Roulette-Wheel Selection to select an individual to the mating pool

        ##############################
        ### YOU'RE CODE GOES HERE ####
        ##############################

        random_stop = random.uniform(0, 1)

        total_fitness = 0
        for f in normalizedFitness:
            total_fitness += f

        fitness_sum = 0
        for i in range(0, len(normalizedFitness)):
            if fitness_sum > random_stop:
                return i
            else:
                fitness_sum += normalizedFitness[i] / total_fitness

        print("Roulette Wheel Select failed. Returns 0 as selection.")
        return 0

    def TournamentSelect(self, fitness, tournamentSelectionParameter, tournamentSize):
        # Use Tournament Selection to select an individual to the mating pool

        ##############################
        ### YOU'RE CODE GOES HERE ####
        ##############################

        first = random.randint(tournamentSize)  # random selection of first individual

        second = random.randint(tournamentSize)  # second individual, guaranteed not the same as the first one
        while second == first:
            second = random.randint(tournamentSize)

        random_chance = random.uniform()

        candidates = [first, second]
        if random_chance < tournamentSelectionParameter:
            return fitness.index(max(candidates))  # return the index of the highest fitness
        else:
            return fitness.index(min(candidates))  # return the index of the lowest fitness

    def Cross(self, chromosome1, chromosome2, crossoverProbability):
        pass

    # Cross the two individuals "in-place"
    # NB! Don't forget to use the crossover probability

    ##############################
    ### YOU'RE CODE GOES HERE ####
    ##############################

    def Mutate(self, chromosome, mutationProbability):
        pass

    # Mutate the individuals "in-place"
    # NB! Don't forget to apply the mutation probability to each bit

    ##############################
    ### YOU'RE CODE GOES HERE ####
    ##############################

    def InsertBestIndividual(self, population, individual, numberOfBestIndividualCopies):
        for i in range(numberOfBestIndividualCopies):
            population[-1 - i] = individual.copy()

    def Step(self):
        if self.populationSize % 2 == 0:
            tempPopulation = np.zeros([self.populationSize, self.numberOfGenes], dtype=int)
        else:
            tempPopulation = np.zeros([self.populationSize + 1, self.numberOfGenes], dtype=int)

        for i in range(0, self.populationSize, 2):
            if self.selectionMethod == 0:
                i1 = self.TournamentSelect(self.fitness, self.tournamentSelectionParameter, self.tournamentSize)
                i2 = self.TournamentSelect(self.fitness, self.tournamentSelectionParameter, self.tournamentSize)
            else:
                i1 = self.RouletteWheelSelect(self.normalizedFitness)
                i2 = self.RouletteWheelSelect(self.normalizedFitness)
            chromosome1 = self.population[i1].copy()
            chromosome2 = self.population[i2].copy()

            self.Cross(chromosome1, chromosome2, self.crossoverProbability)
            self.Mutate(chromosome1, self.mutationProbability)
            self.Mutate(chromosome2, self.mutationProbability)

            tempPopulation[i] = chromosome1
            tempPopulation[i + 1] = chromosome2

        if self.populationSize % 2 != 0:
            tempPopulation = tempPopulation[0:self.populationSize]

        if self.useElitism:
            bestIndividual = self.population[self.bestIndividualIndex]
            self.InsertBestIndividual(tempPopulation, bestIndividual, self.numberOfBestIndividualCopies)
        self.population = tempPopulation
        self.generation += 1
        self.CalculateFitness()

    def Run(self):
        while self.generation < self.numberOfGenerations:
            self.Step()
