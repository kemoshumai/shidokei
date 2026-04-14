use std::collections::HashMap;

#[derive(Debug)]
pub struct RaceRaw {
    pub race_id: u64,
    pub date: String,
    pub venue: Venue,
    pub race_number: u64,
    pub race_name: String,
    pub distance: u64,
    pub track_condition: TrackCondition,
    pub weather: Weather,
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
                    venue: Venue::from_str(record[2].to_string().as_str()),
                    race_number: record[3].parse().unwrap(),
                    race_name: record[4].to_string(),
                    distance: record[5].parse().unwrap(),
                    track_condition: TrackCondition::from_str(record[6].to_string().as_str()),
                    weather: Weather::from_str(record[7].to_string().as_str()),
                    scheduled_time: None, // 2018年の形式ではscheduled_timeが存在しないため、Noneを設定
                    lap_times: Some(record[8].to_string()),
                }
            } else {
                RaceRaw {
                    race_id: record[0].parse().unwrap(),
                    date: record[1].to_string(),
                    venue: Venue::from_str(record[2].to_string().as_str()),
                    race_number: record[3].parse().unwrap(),
                    race_name: record[4].to_string(),
                    scheduled_time: Some(record[5].to_string()),
                    distance: record[6].parse().unwrap(),
                    track_condition: TrackCondition::from_str(record[7].to_string().as_str()),
                    weather: Weather::from_str(record[8].to_string().as_str()),
                    lap_times: None, // 2026年以降の形式ではlap_timesが存在しないため、Noneを設定
                }
            };
            races.insert(race.race_id, race);
        }
        Ok(races)
    }
}

#[derive(Debug, Copy, Clone)]
pub enum Venue {
    Monbetsu = 0,
    Morioka = 1,
    Mizusawa = 2,
    Urawa = 3,
    Funabashi = 4,
    Oi = 5,
    Kawasaki = 6,
    Kanazawa = 7,
    Kasamatsu = 8,
    Nagoya = 9,
    Sonoda = 10,
    Himeji = 11,
    Kochi = 12,
    Saga = 13,
}

impl Venue {
    pub fn from_str(s: &str) -> Self {
        match s {
            "門別" => Self::Monbetsu,
            "盛岡" => Self::Morioka,
            "水沢" => Self::Mizusawa,
            "浦和" => Self::Urawa,
            "船橋" => Self::Funabashi,
            "大井" => Self::Oi,
            "川崎" => Self::Kawasaki,
            "金沢" => Self::Kanazawa,
            "笠松" => Self::Kasamatsu,
            "名古屋" => Self::Nagoya,
            "園田" => Self::Sonoda,
            "姫路" => Self::Himeji,
            "高知" => Self::Kochi,
            "佐賀" => Self::Saga,
            _ => panic!("Unknown venue: {}", s),
        }
    }

    pub fn to_string(&self) -> &'static str {
        match self {
            Self::Monbetsu => "門別",
            Self::Morioka => "盛岡",
            Self::Mizusawa => "水沢",
            Self::Urawa => "浦和",
            Self::Funabashi => "船橋",
            Self::Oi => "大井",
            Self::Kawasaki => "川崎",
            Self::Kanazawa => "金沢",
            Self::Kasamatsu => "笠松",
            Self::Nagoya => "名古屋",
            Self::Sonoda => "園田",
            Self::Himeji => "姫路",
            Self::Kochi => "高知",
            Self::Saga => "佐賀",
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum TrackCondition {
    Unknown = 0,
    Good = 1,
    SlightlyHeavy = 2,
    Heavy = 3,
    Bad = 4,
}

impl TrackCondition {
    pub fn from_str(s: &str) -> Self {
        match s {
            "" => Self::Unknown,
            "良" => Self::Good,
            "稍重" => Self::SlightlyHeavy,
            "重" => Self::Heavy,
            "不良" => Self::Bad,
            _ => panic!("Unknown track condition: {}", s),
        }
    }

    pub fn to_string(&self) -> &'static str {
        match self {
            Self::Unknown => "",
            Self::Good => "良",
            Self::SlightlyHeavy => "稍重",
            Self::Heavy => "重",
            Self::Bad => "不良",
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum Weather {
    Unknown = 0,
    Sunny = 1,
    Cloudy = 2,
    Rain = 3,
    Snow = 4,
    LightSnow = 5,
    LightRain = 6,
}

impl Weather {
    pub fn from_str(s: &str) -> Self {
        match s {
            "" => Self::Unknown,
            "晴" => Self::Sunny,
            "曇" => Self::Cloudy,
            "雨" => Self::Rain,
            "雪" => Self::Snow,
            "小雪" => Self::LightSnow,
            "小雨" => Self::LightRain,
            _ => panic!("Unknown weather: {}", s),
        }
    }

    pub fn to_string(&self) -> &'static str {
        match self {
            Self::Unknown => "",
            Self::Sunny => "晴",
            Self::Cloudy => "曇",
            Self::Rain => "雨",
            Self::Snow => "雪",
            Self::LightSnow => "小雪",
            Self::LightRain => "小雨",
        }
    }
}
