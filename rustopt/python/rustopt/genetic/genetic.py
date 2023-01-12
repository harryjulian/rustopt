import numpy as np

class GeneticAlgorithm:

  def __init__(
    self,
    population_size: int,
    crossover_rate: float,
  ):
    self.population_size = population_size
    self.crossover_rate = crossover_rate