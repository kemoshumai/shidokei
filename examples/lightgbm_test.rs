use std::collections::HashMap;

use shidokei::parser::{races::RaceRaw, results::ResultRaw};

#[derive(Debug)]
struct Input {
    // race
    venue: String,
    distance: u64,
    track_condition: String,
    weather: String,

    // result
    horse_id: u64,     // PRIMARY
    horse_number: u64, // こっちは馬番
    jockey_id: u64,
    trainer_id: u64,
    weight: f64,
    horse_weight: Option<f64>, // 馬が暴れて体重が測れないケース(計不)があるため、Optionにする

    // 予測したい値
    rank: u64,
}

fn main() {
    let (results, races) = load_data();

    println!("全体：{}件", results.len());

    let inputs = results
        .iter()
        .filter_map(|r| {
            let race = races.get(&r.race_id).unwrap();

            let Some(rank) = r.rank else {
                // 出走取り消しなどで順位が存在しないケースは学習データから除外する
                return None;
            };

            Some(Input {
                venue: race.venue.clone(),
                distance: race.distance,
                track_condition: race.track_condition.clone(),
                weather: race.weather.clone(),
                horse_id: r.horse_id,
                horse_number: r.horse_number,
                jockey_id: r.jockey_id,
                trainer_id: r.trainer_id,
                weight: r.weight.unwrap(),
                horse_weight: r.horse_weight,
                rank,
            })
        })
        .collect::<Vec<_>>();

    println!("有効：{}件", inputs.len());

    for i in 0..10 {
        println!("{:?}", inputs[i]);
    }
}

fn load_data() -> (Vec<ResultRaw>, HashMap<u64, RaceRaw>) {
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
