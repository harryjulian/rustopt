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

fn filter_by<I1, I2>(bs: I1, ys: I2) -> impl Iterator<Item=<I2 as Iterator>::Item>
  // Generic function for filtering a vector by a boolean vector.
    where I1: Iterator<Item=bool>,
          I2: Iterator {
      bs.zip(ys).filter(|x| x.0).map(|x| x.1)
}

fn fitness(solution: Vec<bool>, weights: Vec<u64>, max_weight: u64) -> u64 {
  // Eval basic fitness function.
  let items: Vec<_> = filter_by(solution.into_iter(), weights.into_iter()).collect();
  let fitness: u64 = items.iter().sum();

  // Return 0 if constraints are breached
  if fitness < max_weight {
    return fitness
  }
  else {
    return 0
  }
}

fn roulette_selection() -> Vec<T> {
  // Run roulette selection between a given pair of solutions.
}

fn crossover() -> Vec<T> {
  // Crossover genes to create children.
}

fn mutation() -> Vec<T> {
  // Randomly switch bits to induce variance.
}

fn generation() -> Vec<T> {
  // Run an entire generation!

}




 