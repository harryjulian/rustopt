from typing import Tuple, Dict, Union
import numpy as np

class SingleKnapsackProblem:

  def __init__(
    self,
    n_items: int,
    weight_range: Tuple,
    weight_dist_pars: Dict,
    max_weight: Union[int, float],
    seed: int = 42
  ):
    self.n_items = n_items
    self.weight_dist = weight_dist_pars['dist']
    self.weight_dist_pars = {
      key: val for key, val in weight_dist_pars.items() if key != 'dist'
    }
    self.max_weight = max_weight
    self.seed = seed
  
  def _generate_weights(self):
    if self.weight_dist == 'uniform':
      self.weights = np.random.uniform(**self.weight_dist_pars)
    elif self.weight_dist == 'normal':
      self.weights = np.random.normal(**self.weight_dist_pars)
  
  def generate(self):
    return self._generate_weights()
