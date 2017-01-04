import numpy as np


def InitializePopulation(populationSize, numberOfGenes):
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

    k = len(chromosome) / nVariables
    print len(chromosome)
    for i in range(0, nVariables):
        value = 0

        power = -1
        for v in range(k * i, k * i + k):
            value += pow(2, power) * chromosome[v]
            power -= 1


        l = variableRange[i][0]
        u = variableRange[i][1]
        vars[i] = l + (u - l) / (1 - pow(2, -k)) + value

    return vars

varRange = []

nVars = 2

lower = 5
upper = 15

for i in range(0, nVars):
    varRange.append([])
    varRange[i].append(lower)
    varRange[i].append(upper)


result = DecodeChromosome(InitializePopulation(10, 1)[0], 2, varRange)

print(result)

