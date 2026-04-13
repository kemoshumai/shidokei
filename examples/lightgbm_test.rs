use shidokei::parser::races::RaceRaw;

fn main() {
    let file_reader = std::fs::File::open("./data/raw/2025/races.csv").unwrap();
    let races = RaceRaw::from_csv(file_reader).unwrap();
    println!(
        "{:?} / {}",
        races.iter().filter(|r| r.weather == "晴").count(),
        races.len()
    );
}
