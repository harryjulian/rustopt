extern crate rand;
extern crate rand_distr;

use rand::Rng;
use rand::seq::SliceRandom;
use rand_distr::{Bernoulli, Distribution};

// Genetic Algorithm for solving the Discrete Knapsack Problem

fn generate_population(population_size: usize, length: usize) -> Vec<Vec<bool>> {
  // Randomly generate a population of solutions.
  let mut population: Vec<Vec<bool>> = Vec::with_capacity(population_size);
  let dist = Bernoulli::new(0.5).unwrap();

  // For each solution in the population, randomly generate a bytestring
  for solution in 0..population_size {
    let mut bytestring = Vec::with_capacity(length);
    
    for byte in 0..length {
        let v = dist.sample(&mut rand::thread_rng());
        bytestring.push(v)
    }
    population.push(bytestring);
  }
  return population
}

fn eval_fitness(population: &Vec<Vec<bool>>, weights: &Vec<f64>, max_weight: &f64) -> Vec<f64> {
  // Eval basic fitness function.

  let mut fitness_vec: Vec<f64> = Vec::with_capacity(population.len());

  for solution in population.into_iter() {
    let items: Vec<_> = solution.into_iter().zip(weights.into_iter()).filter(|x| *x.0).map(|x| x.1).collect();
    let fitness: f64 = items.iter().map(|x| *x).sum();

    // Return 0 if weight constraint is breached
    if fitness < *max_weight {
      fitness_vec.push(fitness);
    }
    else {
      fitness_vec.push(0.0);
    }
  }
  return fitness_vec
}

fn roulette_selection(fitness: &Vec<f64>) -> Vec<bool> {
  // Run roulette selection across all solutions.
  // Return a boolean mask of those that have been selected.

  let mut selected_idx: Vec<bool> = Vec::new();
  let mut rng = rand::thread_rng();
 
  let population_size: usize = fitness.len();
  let overall_fitness: f64 = fitness.iter().sum();
  let p_choice_i: Vec<f64> = fitness.into_iter().map(|f| f / overall_fitness).collect();

  let mut sum: f64 = 0.0;
  for solution_idx in 0..population_size {
    let r: f64 = rng.gen_range(0.0..0.999);
    sum = sum + r;
    if r < sum {
      selected_idx.push(true);
    }
    else {
      selected_idx.push(false)
    }
  }
  return selected_idx
}

fn select_index_subset(population: &Vec<Vec<bool>>, selection_rate: f64) -> Vec<usize> {
  // Given a selection rate, randomly choose indices from
  // a population for operations to be applied to.

  // Find the top N we need to select from the shuffled indices
  let mut rng = rand::thread_rng();
  let population_size: usize = population.len();
  let population_size_float = population_size as f64;
  let top_n: f64 = selection_rate * population_size_float;
  let top_n = top_n as usize;

  // Shuffle indices, select top n proportion
  let mut indices: Vec<usize> = (0..population_size).collect();
  indices.shuffle(&mut rng);
  let indices = indices[0..top_n].to_vec();

  return indices
}

fn crossover(population: Vec<Vec<bool>>, crossover_rate: f64) -> Vec<Vec<bool>> {
  // Crossover genes to create children.
  // Return crossed-over population.
 
  // Init
  let mut population = population;
  let mut rng = rand::thread_rng();
  let l: usize = population[0].len();
  let rn: usize = l - 1;
  
  // Get index subset
  let indices = select_index_subset(&population, crossover_rate);

  // For each pair of indices, swap bits and overrwrite in population
  for pair in indices.windows(2).step_by(2) {
    let idx1: usize = pair[0];
    let idx2: usize = pair[1];
    
    let mut whole: Vec<bool> = Vec::new();
    let p1 = &population[idx1];
    let p2 = &population[idx2];
    for i in p1.into_iter() {
        whole.push(*i);
    }
    for j in p2.into_iter() {
        whole.push(*j);
    }
    
    // Randomly select where we start the 2 bit change
    let bit_location: usize = rng.gen_range(0..l);
    
    // Crossover bits
    for bit_1 in bit_location..bit_location+1 {
         let bit_2: usize = bit_1 + rn;
         whole.swap(bit_1, bit_2);
     }
    let (p1, p2) = whole.split_at(rn+1);
    population[pair[0]] = p2.to_vec();
    population[pair[1]] = p1.to_vec();
  }
  return population
}

fn mutation(population: Vec<Vec<bool>>, mutation_rate: f64) -> Vec<Vec<bool>> {
  // Randomly switch bits to induce variance.

  // Initialise
  let mut population = population;
  let mut rng = rand::thread_rng();
  let l: usize = population[0].len();

  // Randomly select indices
  let indices = select_index_subset(&population, mutation_rate);

  // For all indices, flip a randomly select bit
  for idx in indices.into_iter() {
    let mutant = &mut population[idx];
    let changed_bit: usize = rng.gen_range(0..l);
    mutant[changed_bit] ^=mutant[changed_bit]; // Flip bit
    population[idx] = mutant.to_vec();
  }
  return population
}

pub struct GeneticAlgorithm {
  pub n_generations: usize,
  pub population_size: usize,
  pub length: usize,
  pub weights: Vec<f64>,
  pub max_weight: f64,
  pub crossover_rate: f64,
  pub mutation_rate: f64,
}

impl GeneticAlgorithm {

  pub fn new(n_generations: usize,
    population_size: usize,
    length: usize,
    weights: Vec<f64>,
    max_weight: f64,
    crossover_rate: f64,
    mutation_rate: f64
  ) -> GeneticAlgorithm {
    
    // Ensure solution size and N weights are the same
    let weights_len: usize = weights.len();
    assert_eq!(weights_len, length);

    // Then initialise
    GeneticAlgorithm {
      n_generations,
      population_size,
      length,
      weights,
      max_weight,
      crossover_rate,
      mutation_rate
    }
  }

  pub fn run(self) -> (f64, Vec<bool>) {

    // Initialise Population
    let mut population: Vec<Vec<bool>> = generate_population(
      self.population_size, self.length
    );
    assert_eq!(population.len(), self.population_size);

    let fitness: Vec<f64> = eval_fitness(
      &population, &self.weights, &self.max_weight
    );

    // Iterate over all generations
    for _gen in 0..self.n_generations {

      // Perform Selection
      let selected_idx = roulette_selection(&fitness);
      let unselected_idx: Vec<bool> = selected_idx.clone().into_iter().map(|b| if b {false} else {true}).collect();
      assert_ne!(selected_idx, unselected_idx);

      let mut selected_population = population.clone();
      let mut unselected_population = population.clone();
      selected_population.retain(|_| *selected_idx.iter().next().unwrap());
      unselected_population.retain(|_| *unselected_idx.iter().next().unwrap());

      // Perform crossover
      unselected_population = crossover(unselected_population, self.crossover_rate);

      // Perform mutation
      unselected_population = mutation(unselected_population, self.mutation_rate);

      // Get population back together.
      population = [&selected_population[..], &unselected_population[..]].concat();
      assert_eq!(population.len(), self.population_size);
      
      // Eval fitness again
      let fitness: Vec<f64> = eval_fitness(
        &population, &self.weights, &self.max_weight
      );
      println!("Generation best fitness: {:?}", fitness.iter().cloned().fold(0./0., f64::max));
    }

    // Output best solution!
    let best_solution_fitness: f64 = fitness.iter().cloned().fold(0./0., f64::max);
    let best_solution_idx = fitness.iter().position(|x| *x == best_solution_fitness).unwrap();
    let best_solution = &population[best_solution_idx];
    return (best_solution_fitness, best_solution.to_vec());
  }
}
