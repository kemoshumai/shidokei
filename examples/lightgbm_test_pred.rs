use std::collections::{HashMap, HashSet};

use lightgbm3::{Booster, Dataset};
use serde_json::json;
use shidokei::parser::{
    races::{RaceRaw, TrackCondition, Venue, Weather},
    results::ResultRaw,
};

#[derive(Debug)]
struct Input {
    // race
    venue: Venue,
    distance: u64,
    track_condition: TrackCondition,
    weather: Weather,

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

impl Input {
    fn to_feature_vector(&self) -> Vec<f64> {
        vec![
            self.venue as u32 as f64,
            self.distance as f64,
            u32_zero_as_nan(self.track_condition as u32),
            u32_zero_as_nan(self.weather as u32),
            self.horse_id as f64,
            self.horse_number as f64,
            self.jockey_id as f64,
            self.trainer_id as f64,
            self.weight as f64,
            self.horse_weight.unwrap_or(f64::NAN), // 馬が暴れて体重が測れないケースはNaNにする
        ]
    }
}

fn main() {
    let horse_ids = [
        30087402996, // ミチ
        30005401896, // スイートロータス
        30013406896, // クールブリーズ
        30068405686, // レジーナチェリ
        30058408356, // バジルフレイバー
        30077400466, // ヨシノローズ
        30035405586, // ヒンナ
        30055402196, // タール
        30048400196, // ハクアイ
        30026404896, // ヴァルチャースター
    ];

    let jockey_ids = [
        30878, 31231, 31319, 80171, 31082, 31241, 31352, 31267, 31323, 31323,
    ];

    let trainer_ids = [
        11315, 11495, 11371, 11353, 11175, 11329, 11468, 11329, 11301, 11371,
    ];
    let weights = [444, 437, 473, 444, 468, 464, 464, 524, 449, 456];

    let inputs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map(|i| Input {
        venue: Venue::Oi,
        distance: 1600,
        track_condition: TrackCondition::Good,
        weather: Weather::Sunny,
        horse_id: horse_ids[i],
        horse_number: i as u64 + 1,
        jockey_id: jockey_ids[i],
        trainer_id: trainer_ids[i],
        weight: weights[i] as f64,
        horse_weight: None,
        rank: 1, // ダミーの順位
    });

    for input in inputs {
        let predicted_rank = predict_rank(&input);
        println!(
            "Horse ID: {}, Predicted Rank: {}",
            input.horse_id, predicted_rank
        );
    }

    println!("Done.");
}

fn predict_rank(input: &Input) -> f64 {
    let bst = Booster::from_file("./model.lgb").unwrap();
    let features = input.to_feature_vector();
    let n_features = features.len();
    let y_pred = bst
        .predict_with_params(&features, n_features as i32, true, "num_threads=1")
        .unwrap()[0];

    y_pred
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

fn u32_zero_as_nan(value: u32) -> f64 {
    if value == 0 { f64::NAN } else { value as f64 }
}
