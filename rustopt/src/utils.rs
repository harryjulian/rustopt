// Various Utility functions.

fn filter_by<I1, I2>(bs: I1, ys: I2) -> impl Iterator<Item=<I2 as Iterator>::Item>
  // Generic function for filtering a vector by a boolean vector.
    where I1: Iterator<Item=bool>,
          I2: Iterator {
      bs.zip(ys).filter(|x| x.0).map(|x| x.1)
}
