mod genetic;
use genetic::GeneticAlgorithm;
use pyo3::prelude::*;

#[pyfunction]
fn run_genetic_algorithm(
    n_generations: usize,
    population_size: usize,
    length: usize,
    weights: Vec<f64>,
    max_weight: f64,
    crossover_rate: f64,
    mutation_rate: f64,
) -> PyResult<(f64, Vec<bool>)> {
    let ga = GeneticAlgorithm {
        n_generations: n_generations,
        population_size: population_size,
        length: length,
        weights: weights,
        max_weight: max_weight,
        crossover_rate: crossover_rate,
        mutation_rate: mutation_rate
    };
    let res = ga.run();
    return Ok(res) 
}

#[pymodule]
fn rustopt(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_genetic_algorithm, m)?)?;
    Ok(())
}