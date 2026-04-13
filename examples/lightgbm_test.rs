use shidokei::parser::results::ResultRaw;

fn main() {
    let file_reader = std::fs::File::open("./data/raw/2025/results.csv").unwrap();
    let results = ResultRaw::from_csv(file_reader).unwrap();
    println!(
        "{:?} / {}",
        results.iter().filter(|r| r.horse_name == "モンド").count(),
        results.len()
    );
}
