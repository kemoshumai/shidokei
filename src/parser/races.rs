use std::collections::HashMap;

#[derive(Debug)]
pub struct RaceRaw {
    pub race_id: u64,
    pub date: String,
    pub venue: String,
    pub race_number: u64,
    pub race_name: String,
    pub distance: u64,
    pub track_condition: String,
    pub weather: String,
    pub scheduled_time: Option<String>,
    pub lap_times: Option<String>,
}

impl RaceRaw {
    pub fn from_csv<R>(rdr: R) -> Result<HashMap<u64, Self>, csv::Error>
    where
        R: std::io::Read,
    {
        let mut rdr = csv::ReaderBuilder::new().has_headers(true).from_reader(rdr);
        let mut races = HashMap::new();

        let header = rdr.headers()?.clone();
        let expected_2018_format = header.iter().find(|p| p.ends_with("lap_times")).is_some();

        for result in rdr.records() {
            let record = result?;
            let race = if expected_2018_format {
                RaceRaw {
                    race_id: record[0].parse().unwrap(),
                    date: record[1].to_string(),
                    venue: record[2].to_string(),
                    race_number: record[3].parse().unwrap(),
                    race_name: record[4].to_string(),
                    distance: record[5].parse().unwrap(),
                    track_condition: record[6].to_string(),
                    weather: record[7].to_string(),
                    scheduled_time: None, // 2018年の形式ではscheduled_timeが存在しないため、Noneを設定
                    lap_times: Some(record[8].to_string()),
                }
            } else {
                RaceRaw {
                    race_id: record[0].parse().unwrap(),
                    date: record[1].to_string(),
                    venue: record[2].to_string(),
                    race_number: record[3].parse().unwrap(),
                    race_name: record[4].to_string(),
                    scheduled_time: Some(record[5].to_string()),
                    distance: record[6].parse().unwrap(),
                    track_condition: record[7].to_string(),
                    weather: record[8].to_string(),
                    lap_times: None, // 2026年以降の形式ではlap_timesが存在しないため、Noneを設定
                }
            };
            races.insert(race.race_id, race);
        }
        Ok(races)
    }
}
