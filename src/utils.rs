use std::collections::HashMap;

use crate::parser::{races::RaceRaw, results::ResultRaw};

pub fn load_data() -> (Vec<ResultRaw>, HashMap<u64, RaceRaw>) {
    let mut all_results = Vec::new();
    let mut all_races = HashMap::new();

    for year in 2018..=2025 {
        println!("Year: {}", year);

        let results = format!("./data/raw/{}/results.csv", year);
        let results = get_results(results.as_str());

        let races = format!("./data/raw/{}/races.csv", year);
        let races = get_races(races.as_str());

        all_results.extend(results);
        all_races.extend(races);
    }

    (all_results, all_races)
}

fn get_results(path: &str) -> Vec<ResultRaw> {
    let file_reader = std::fs::File::open(&path).unwrap();
    ResultRaw::from_csv(file_reader).unwrap()
}

fn get_races(path: &str) -> HashMap<u64, RaceRaw> {
    let file_reader = std::fs::File::open(&path).unwrap();
    RaceRaw::from_csv(file_reader).unwrap()
}

pub fn u32_zero_as_nan(value: u32) -> f64 {
    if value == 0 { f64::NAN } else { value as f64 }
}
