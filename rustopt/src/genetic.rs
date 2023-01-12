use rand::Rng;
use rand_distr::{Bernoulli, Distribution};
mod utils;

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

fn fitness(solution: Vec<bool>, weights: Vec<u64>, max_weight: u64) -> u64 {
  // Eval basic fitness function.
  let items: Vec<_> = utils::filter_by(solution.into_iter(), weights.into_iter()).collect();
  let fitness: u64 = items.iter().sum();

  // Return 0 if weight constraint is breached
  if fitness < max_weight {
    return fitness
  }
  else {
    return 0
  }
}

fn roulette_selection(fitness: Vec<f64>) -> Vec<bool> {
  // Run roulette selection across all solutions, return
  // a boolean mask of those that have been selected.

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

fn crossover(population: Vec<Vec<bool>>, crossover_rate: f64) -> Vec<Vec<bool>> {
  // Crossover genes to create children.
 
  // Generate remaining indices
  let mut population = population;
  let mut rng = rand::thread_rng();
  let l: usize = population[0].len() - 1;
  let population_size: usize = population.len();
  let top_n: usize = crossover_rate * population_size;

  // Shuffle indices
  let indices: Vec<i8> = (0..population_size).collect();
  indices.shuffle(&mut rng);

  // For each pair of indices, swap bits and overrwrite in population
  for pair in indices.into_iter().window(2).step_by(2) {
    let bit_location: i8 = rng.gen_range(0..l);
    let p1: Vec<bool> = population[pair[0]];
    let p2: Vec<bool> = population[pair[1]];
    let p1_swap = p1[bit_location..bit_location+1];
    let p2_swap = p1[bit_location..bit_location+1];
    p1[bit_location..bit_location+1] = p2_swap;
    p2[bit_location..bit_location+1] = p1_swap;

    population[pair[0]] = p1;
    population[pair[1]] = p2;
  }
  return population
}

fn mutation(population: Vec<Vec<bool>>, mutation_rate: f64) -> Vec<Vec<bool>> {
  // Randomly switch bits to induce variance.
}

fn generation(population: Vec<Vec<bool>>) -> Vec<T> {
  // Run an entire generation!

}




 