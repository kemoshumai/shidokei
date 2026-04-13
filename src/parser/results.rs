#[derive(Debug)]
pub struct ResultRaw {
    pub race_id: u64,
    pub horse_id: u64,
    pub horse_name: String,
    pub horse_number: u64,
    pub frame_number: u64,
    pub jockey_id: u64,
    pub trainer_id: u64,
    pub weight: Option<f64>,
    pub horse_weight: Option<f64>, // 後ろ8項目が全部Emptyになるケースがある
    pub rank: Option<u64>,
    pub time: Option<String>,
    pub last_3f: Option<f64>,
    pub popularity: Option<u64>,
    pub odds: Option<f64>,
    pub corner3: Option<u64>,
    pub corner4: Option<u64>,
}

impl ResultRaw {
    pub fn from_csv<R>(rdr: R) -> Result<Vec<Self>, csv::Error>
    where
        R: std::io::Read,
    {
        let mut rdr = csv::ReaderBuilder::new().has_headers(true).from_reader(rdr);
        let mut results = Vec::new();

        // race_id,horse_id,horse_name,horse_number,frame_number,jockey_id,trainer_id,
        // weight,horse_weight,rank,time,last_3f,popularity,odds,corner3,corner4
        for record in rdr.records() {
            let record = record?;

            let result = ResultRaw {
                race_id: record[0].parse().unwrap(),
                horse_id: record[1].parse().unwrap(),
                horse_name: record[2].to_string(),
                horse_number: record[3].parse().unwrap(),
                frame_number: record[4].parse().unwrap(),
                jockey_id: record[5].parse().unwrap(),
                trainer_id: record[6].parse().unwrap(),
                weight: record[7].parse().ok(),
                horse_weight: record[8].parse().ok(),
                rank: record[9].parse().ok(),
                time: record[10].parse().ok(),
                last_3f: record[11].parse().ok(),
                popularity: record[12].parse().ok(),
                odds: record[13].parse().ok(),
                corner3: record[14].parse().ok(),
                corner4: record[15].parse().ok(),
            };

            results.push(result);
        }

        Ok(results)
    }
}
