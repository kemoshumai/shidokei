use std::cmp::Ordering;
use std::collections::HashMap;

use rankit::{LambdaRankParams, LambdaRankTrainer, compute_lambdarank_gradients, ndcg_at_k};
use serde_json::{Value, json};
use shidokei::parser::{races::RaceRaw, results::ResultRaw};
use shidokei::utils::{load_data, u32_zero_as_nan};

const VENUE_DIM: usize = 16;
const TRACK_COND_DIM: usize = 8;
const WEATHER_DIM: usize = 8;

const HORSE_ID_BUCKETS: usize = 4096;
const JOCKEY_ID_BUCKETS: usize = 2048;
const TRAINER_ID_BUCKETS: usize = 2048;

#[derive(Debug, Clone)]
struct HorseInput {
    race_id: u64,
    rank: u64, // 1 = best
    features: Vec<f32>,
}

#[derive(Debug, Clone)]
struct FeatureScaler {
    means: Vec<f32>,
    stds: Vec<f32>,
}

#[derive(Debug, Clone)]
struct RankingModel {
    weights: Vec<f32>,
    scaler: FeatureScaler,
}

fn hash_bucket_u64(id: u64, buckets: usize, seed: u64) -> usize {
    // splitmix64-like hash
    let mut x = id ^ seed;
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^= x >> 31;
    (x as usize) % buckets
}

fn one_hot_from_u32(value: u32, dim: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; dim];
    if value > 0 {
        let idx = (value as usize - 1).min(dim - 1);
        v[idx] = 1.0;
    }
    v
}

fn id_hashed_one_hot(id: u64, buckets: usize, seed: u64) -> Vec<f32> {
    let mut v = vec![0.0_f32; buckets];
    let idx = hash_bucket_u64(id, buckets, seed);
    v[idx] = 1.0;
    v
}

fn build_inputs(results: &[ResultRaw], races: &HashMap<u64, RaceRaw>) -> Vec<HorseInput> {
    // まず race ごとにまとめる（レース内相対特徴を作るため）
    let mut grouped_raw: HashMap<u64, Vec<&ResultRaw>> = HashMap::new();
    for r in results {
        if r.rank.is_some() && races.contains_key(&r.race_id) {
            grouped_raw.entry(r.race_id).or_default().push(r);
        }
    }

    let mut out = Vec::new();

    for (race_id, rows) in grouped_raw {
        let Some(race) = races.get(&race_id) else {
            continue;
        };

        // レース内相対特徴に使う数値（欠損は除外して平均・std）
        let weights: Vec<f32> = rows
            .iter()
            .map(|r| r.weight.unwrap_or(f64::NAN) as f32)
            .collect();
        let horse_weights: Vec<f32> = rows
            .iter()
            .map(|r| r.horse_weight.unwrap_or(f64::NAN) as f32)
            .collect();
        let horse_numbers: Vec<f32> = rows.iter().map(|r| r.horse_number as f32).collect();

        let race_mean_std = |xs: &[f32]| -> (f32, f32) {
            let valid: Vec<f32> = xs.iter().copied().filter(|x| !x.is_nan()).collect();
            if valid.is_empty() {
                return (0.0, 1.0);
            }
            let mean = valid.iter().sum::<f32>() / valid.len() as f32;
            let var = valid
                .iter()
                .map(|x| {
                    let d = *x - mean;
                    d * d
                })
                .sum::<f32>()
                / (valid.len().max(2) as f32);
            let std = var.sqrt().max(1e-6);
            (mean, std)
        };

        let (w_mean, w_std) = race_mean_std(&weights);
        let (hw_mean, hw_std) = race_mean_std(&horse_weights);
        let (hn_mean, hn_std) = race_mean_std(&horse_numbers);

        for r in rows {
            let Some(rank) = r.rank else { continue };

            let w = r.weight.unwrap_or(f64::NAN) as f32;
            let hw = r.horse_weight.unwrap_or(f64::NAN) as f32;
            let hn = r.horse_number as f32;

            let w_delta = if w.is_nan() { f32::NAN } else { w - w_mean };
            let w_z = if w.is_nan() {
                f32::NAN
            } else {
                (w - w_mean) / w_std
            };

            let hw_delta = if hw.is_nan() { f32::NAN } else { hw - hw_mean };
            let hw_z = if hw.is_nan() {
                f32::NAN
            } else {
                (hw - hw_mean) / hw_std
            };

            let hn_delta = hn - hn_mean;
            let hn_z = (hn - hn_mean) / hn_std;

            let mut feature_vec = Vec::<f32>::new();

            // one-hot categorical (global, fixed dims)
            feature_vec.extend(one_hot_from_u32(race.venue as u32, VENUE_DIM));
            feature_vec.extend(one_hot_from_u32(
                u32_zero_as_nan(race.track_condition as u32) as u32,
                TRACK_COND_DIM,
            ));
            feature_vec.extend(one_hot_from_u32(
                u32_zero_as_nan(race.weather as u32) as u32,
                WEATHER_DIM,
            ));

            // numeric base
            feature_vec.push(race.distance as f32);
            feature_vec.push(hn);
            feature_vec.push(w);
            feature_vec.push(hw);

            // race-relative
            feature_vec.push(hn_delta);
            feature_vec.push(hn_z);
            feature_vec.push(w_delta);
            feature_vec.push(w_z);
            feature_vec.push(hw_delta);
            feature_vec.push(hw_z);

            // hashed ID one-hot
            feature_vec.extend(id_hashed_one_hot(
                r.horse_id,
                HORSE_ID_BUCKETS,
                0x1234_5678_9abc_def0,
            ));
            feature_vec.extend(id_hashed_one_hot(
                r.jockey_id as u64,
                JOCKEY_ID_BUCKETS,
                0x2234_5678_9abc_def0,
            ));
            feature_vec.extend(id_hashed_one_hot(
                r.trainer_id as u64,
                TRAINER_ID_BUCKETS,
                0x3234_5678_9abc_def0,
            ));

            out.push(HorseInput {
                race_id: r.race_id,
                rank,
                features: feature_vec,
            });
        }
    }

    out
}

fn rank_to_relevance(rank: u64, race_size: usize) -> f32 {
    // レース頭数依存：上位ほど大きく、頭数が多いレースで勾配が薄まらないようにスケール
    if rank == 0 || race_size == 0 {
        return 0.0;
    }
    let pos = rank.min(race_size as u64) as f32;
    let n = race_size as f32;

    // [0,1] に正規化された gain（1位=1, 最下位≈0）
    let rel01 = ((n + 1.0) - pos) / n;
    // 少し強調（平方）
    rel01 * rel01 * 3.0
}

fn compute_scaler(groups: &[Vec<HorseInput>]) -> FeatureScaler {
    let feat_dim = groups[0][0].features.len();
    let mut sums = vec![0.0_f64; feat_dim];
    let mut counts = vec![0_u64; feat_dim];

    for g in groups {
        for h in g {
            for (j, x) in h.features.iter().enumerate() {
                if !x.is_nan() {
                    sums[j] += *x as f64;
                    counts[j] += 1;
                }
            }
        }
    }

    let means: Vec<f32> = (0..feat_dim)
        .map(|j| {
            if counts[j] == 0 {
                0.0
            } else {
                (sums[j] / counts[j] as f64) as f32
            }
        })
        .collect();

    let mut var_sums = vec![0.0_f64; feat_dim];
    for g in groups {
        for h in g {
            for (j, x) in h.features.iter().enumerate() {
                if !x.is_nan() {
                    let d = *x as f64 - means[j] as f64;
                    var_sums[j] += d * d;
                }
            }
        }
    }

    let stds: Vec<f32> = (0..feat_dim)
        .map(|j| {
            if counts[j] <= 1 {
                1.0
            } else {
                let v = (var_sums[j] / counts[j] as f64) as f32;
                let s = v.sqrt();
                if s < 1e-6 { 1.0 } else { s }
            }
        })
        .collect();

    FeatureScaler { means, stds }
}

fn apply_scaler_in_place(groups: &mut [Vec<HorseInput>], scaler: &FeatureScaler) {
    for g in groups {
        for h in g {
            for j in 0..h.features.len() {
                let x = h.features[j];
                let xv = if x.is_nan() { scaler.means[j] } else { x };
                h.features[j] = (xv - scaler.means[j]) / scaler.stds[j];
            }
        }
    }
}

fn dot(weights: &[f32], features: &[f32]) -> f32 {
    weights
        .iter()
        .zip(features.iter())
        .map(|(w, x)| w * x)
        .sum()
}

fn ndcg_from_scores(scores: &[f32], relevance: &[f32]) -> f32 {
    let mut indices: Vec<usize> = (0..scores.len()).collect();
    indices.sort_unstable_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(Ordering::Equal));
    let ordered_rel: Vec<f32> = indices.iter().map(|&i| relevance[i]).collect();
    ndcg_at_k(&ordered_rel, None, true).unwrap_or(0.0)
}

fn evaluate_mean_ndcg(groups: &[Vec<HorseInput>], weights: &[f32]) -> f32 {
    if groups.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0_f32;
    for g in groups {
        let scores: Vec<f32> = g.iter().map(|h| dot(weights, &h.features)).collect();
        let rel: Vec<f32> = g
            .iter()
            .map(|h| rank_to_relevance(h.rank, g.len()))
            .collect();
        sum += ndcg_from_scores(&scores, &rel);
    }
    sum / groups.len() as f32
}

fn to_json(model: &RankingModel) -> Value {
    json!({
        "weights": model.weights,
        "scaler": {
            "means": model.scaler.means,
            "stds": model.scaler.stds
        }
    })
}

fn from_json(v: &Value) -> RankingModel {
    let weights: Vec<f32> = v["weights"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect();

    let means: Vec<f32> = v["scaler"]["means"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect();

    let stds: Vec<f32> = v["scaler"]["stds"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect();

    RankingModel {
        weights,
        scaler: FeatureScaler { means, stds },
    }
}

fn save_model(path: &str, model: &RankingModel) {
    let text = serde_json::to_string_pretty(&to_json(model)).unwrap();
    if let Some(parent) = std::path::Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).unwrap();
        }
    }
    std::fs::write(path, text).unwrap();
}

fn load_model(path: &str) -> RankingModel {
    let text = std::fs::read_to_string(path).unwrap();
    let v: Value = serde_json::from_str(&text).unwrap();
    from_json(&v)
}

fn split_train_valid(
    mut groups: Vec<Vec<HorseInput>>,
    valid_ratio: f32,
) -> (Vec<Vec<HorseInput>>, Vec<Vec<HorseInput>>) {
    groups.sort_by_key(|g| g[0].race_id);
    let n = groups.len();
    let valid_n = ((n as f32) * valid_ratio).round() as usize;
    let valid_n = valid_n.min(n.saturating_sub(1));
    let train_n = n - valid_n;
    let valid = groups.split_off(train_n);
    (groups, valid)
}

fn train_and_save(model_path: &str) {
    let (results, races) = load_data();
    let all_inputs = build_inputs(&results, &races);

    let mut grouped: HashMap<u64, Vec<HorseInput>> = HashMap::new();
    for row in all_inputs {
        grouped.entry(row.race_id).or_default().push(row);
    }

    let mut race_groups: Vec<Vec<HorseInput>> = grouped
        .into_values()
        .filter(|g| {
            if g.len() < 2 {
                return false;
            }
            let min_rank = g.iter().map(|x| x.rank).min().unwrap_or(1);
            let max_rank = g.iter().map(|x| x.rank).max().unwrap_or(1);
            min_rank != max_rank
        })
        .collect();

    race_groups.sort_by_key(|g| g[0].race_id);

    if race_groups.len() < 10 {
        println!("有効なレース数が少なすぎます: {}", race_groups.len());
        return;
    }

    let (mut train_groups, mut valid_groups) = split_train_valid(race_groups, 0.2);
    let scaler = compute_scaler(&train_groups);
    apply_scaler_in_place(&mut train_groups, &scaler);
    apply_scaler_in_place(&mut valid_groups, &scaler);

    let feat_dim = train_groups[0][0].features.len();
    let mut weights = vec![0.0_f32; feat_dim];

    let params = LambdaRankParams {
        sigma: 1.0,
        query_normalization: true,
        cost_sensitivity: true,
        score_normalization: false,
        exponential_gain: true,
    };
    let trainer = LambdaRankTrainer::new(params);

    let demo = &train_groups[0];
    let demo_scores: Vec<f32> = demo.iter().map(|x| dot(&weights, &x.features)).collect();
    let demo_relevance: Vec<f32> = demo
        .iter()
        .map(|x| rank_to_relevance(x.rank, demo.len()))
        .collect();
    let mut demo_indices: Vec<usize> = (0..demo_scores.len()).collect();
    demo_indices.sort_unstable_by(|&a, &b| {
        demo_scores[b]
            .partial_cmp(&demo_scores[a])
            .unwrap_or(Ordering::Equal)
    });
    let demo_ordered_rel: Vec<f32> = demo_indices.iter().map(|&i| demo_relevance[i]).collect();
    let demo_ndcg = ndcg_at_k(&demo_ordered_rel, None, true).unwrap_or(0.0);

    println!("--- Demo train race (race_id={}) ---", demo[0].race_id);
    println!("Current ranking by score: {:?}", demo_indices);
    println!("Relevance in that order:  {:?}", demo_ordered_rel);
    println!("NDCG (full list):         {:.4}", demo_ndcg);

    let demo_lambdas =
        compute_lambdarank_gradients(&demo_scores, &demo_relevance, params, None).unwrap();
    println!("LambdaRank gradients:");
    for (i, (&s, &l)) in demo_scores.iter().zip(demo_lambdas.iter()).enumerate() {
        let rel = demo_relevance[i];
        let direction = if l > 0.0 {
            "push DOWN"
        } else if l < 0.0 {
            "push UP"
        } else {
            "no change"
        };
        println!("  horse_{i}: score={s:+.5}, rel={rel:.3}, lambda={l:+.6} ({direction})");
    }

    let lr = 0.01_f32;
    let l2 = 1e-4_f32;
    let grad_clip = 5.0_f32;
    let epochs = 100;
    let patience = 10;
    let min_delta = 1e-4_f32;
    let mut stale = 0usize;

    let mut best_valid = -1.0_f32;
    let mut best_weights = weights.clone();

    println!(
        "train_races={}, valid_races={}, feat_dim={}",
        train_groups.len(),
        valid_groups.len(),
        feat_dim
    );

    for epoch in 0..epochs {
        let mut train_before = 0.0_f32;
        let mut train_after = 0.0_f32;

        for group in &train_groups {
            let scores: Vec<f32> = group.iter().map(|x| dot(&weights, &x.features)).collect();
            let relevance: Vec<f32> = group
                .iter()
                .map(|x| rank_to_relevance(x.rank, group.len()))
                .collect();

            train_before += ndcg_from_scores(&scores, &relevance);

            let batch_lambdas = trainer
                .compute_gradients_batch(&vec![scores.clone()], &vec![relevance.clone()], None)
                .unwrap();
            let lambdas = &batch_lambdas[0];

            let mut grad_w = vec![0.0_f32; feat_dim];
            for (i, horse) in group.iter().enumerate() {
                let lam = lambdas[i].clamp(-1.0, 1.0);
                for (j, x) in horse.features.iter().enumerate() {
                    grad_w[j] += lam * *x;
                }
            }

            for j in 0..feat_dim {
                grad_w[j] += l2 * weights[j];
                grad_w[j] = grad_w[j].clamp(-grad_clip, grad_clip);
                weights[j] -= lr * grad_w[j];
            }

            let updated_scores: Vec<f32> =
                group.iter().map(|x| dot(&weights, &x.features)).collect();
            train_after += ndcg_from_scores(&updated_scores, &relevance);
        }

        let train_n = train_groups.len() as f32;
        let train_before_mean = train_before / train_n;
        let train_after_mean = train_after / train_n;
        let valid_mean = evaluate_mean_ndcg(&valid_groups, &weights);

        let improved = valid_mean > best_valid + min_delta;
        if improved {
            best_valid = valid_mean;
            best_weights = weights.clone();
            stale = 0;
        } else {
            stale += 1;
        }

        println!(
            "epoch {:>3}: train {:.4} -> {:.4}, valid {:.4} (best {:.4}){}",
            epoch + 1,
            train_before_mean,
            train_after_mean,
            valid_mean,
            best_valid,
            if improved { " *" } else { "" }
        );

        if stale >= patience {
            println!(
                "early stopping at epoch {} (patience={}, min_delta={})",
                epoch + 1,
                patience,
                min_delta
            );
            break;
        }
    }

    let model = RankingModel {
        weights: best_weights,
        scaler,
    };
    save_model(model_path, &model);
    println!("saved model: {}", model_path);

    println!("\n--- Validation preview (top 3 races) ---");
    for group in valid_groups.iter().take(3) {
        let mut scored: Vec<(u64, u64, f32)> = group
            .iter()
            .map(|h| (h.race_id, h.rank, dot(&model.weights, &h.features)))
            .collect();
        scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal));

        println!("race_id={}", group[0].race_id);
        for (pred_pos, (_race_id, true_rank, score)) in scored.iter().enumerate() {
            println!(
                "  pred_rank={:>2}, true_rank={:>2}, score={:+.5}",
                pred_pos + 1,
                true_rank,
                score
            );
        }
    }

    println!("\nDone. (rankit-based ranking example, no LightGBM)");
}

fn predict_only(model_path: &str) {
    let model = load_model(model_path);
    let (results, races) = load_data();
    let all_inputs = build_inputs(&results, &races);

    let mut grouped: HashMap<u64, Vec<HorseInput>> = HashMap::new();
    for row in all_inputs {
        grouped.entry(row.race_id).or_default().push(row);
    }

    let mut race_groups: Vec<Vec<HorseInput>> =
        grouped.into_values().filter(|g| g.len() >= 2).collect();

    race_groups.sort_by_key(|g| g[0].race_id);
    apply_scaler_in_place(&mut race_groups, &model.scaler);

    println!("loaded model: {}", model_path);
    println!("--- Prediction preview (latest 3 races) ---");

    for group in race_groups.iter().rev().take(3) {
        let mut scored: Vec<(u64, u64, f32)> = group
            .iter()
            .map(|h| (h.race_id, h.rank, dot(&model.weights, &h.features)))
            .collect();
        scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal));

        println!("race_id={}", group[0].race_id);
        for (pred_pos, (_race_id, true_rank, score)) in scored.iter().enumerate() {
            println!(
                "  pred_rank={:>2}, true_rank={:>2}, score={:+.5}",
                pred_pos + 1,
                true_rank,
                score
            );
        }
    }
}

fn main() {
    let model_path = "./model/rankit_model.json";
    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--predict") {
        predict_only(model_path);
    } else {
        train_and_save(model_path);
    }
}
