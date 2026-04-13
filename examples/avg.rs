use shidokei::parser::{races::RaceRaw, results::ResultRaw};

fn main() {
    let (results, _races) = load_data();

    let (rank_sum, rank_count) = results
        .iter()
        .filter(|r| r.horse_name == "モンド")
        .filter_map(|r| r.rank)
        .fold((0_u64, 0_u64), |(sum, cnt), rank| (sum + rank, cnt + 1));

    let rank_avg = rank_sum as f64 / rank_count as f64;

    println!("{}", rank_avg);
}

fn load_data() -> (Vec<ResultRaw>, Vec<RaceRaw>) {
    let mut all_results = Vec::new();
    let mut all_races = Vec::new();

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

fn get_races(path: &str) -> Vec<RaceRaw> {
    let file_reader = std::fs::File::open(&path).unwrap();
    RaceRaw::from_csv(file_reader).unwrap()
}
