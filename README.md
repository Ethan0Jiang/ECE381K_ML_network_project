# Fake News Detection — Class Project README

A structured, implementation-ready plan for detecting fake news using a user–news sharing network, trustworthiness scoring, feature engineering, and a simple neural model.

## Overview

- Task: Classify news as fake or real using the sharing behavior of users on social media.
- Key idea: Users’ historical sharing patterns inform a trustworthiness score. Aggregating these scores over the sharers of a news item yields features for a classifier.
- Approach: Build a bipartite graph (Users ↔ News), compute user trustworthiness, engineer news-level features, and train a small neural network.

## Data

- Source: PolitiFact-style datasets with fake and real news CSVs (e.g., politifact_fake.csv, politifact_real.csv).
- Required fields:
  - id: Unique identifier for the news item.
  - title: News title.
  - tweet_ids: Tab-separated list of user post identifiers that shared the news.

## Workflow

1. Data preprocessing
   - Load fake and real CSVs.
   - Extract user–news pairs by cleaning tweet IDs.
   - Standardize IDs (e.g., user_*, news_*).

2. Build user–news sharing network
   - Graph type: Bipartite (Users ↔ News).
   - Nodes:
     - User nodes: type=user; track counts of fake vs. real shares; derive total_shares and trustworthiness.
     - News nodes: type=news; label (fake/real); title; share_count; avg_user_trust.
   - Edges:
     - User → News with relationship=shared and optional weight.

3. User trustworthiness
   - Objective: Score users based on historical sharing of real vs. fake.
   - Method: Weighted ratio with confidence adjustment toward neutral for sparse users.
   - Intuition:
     - Users with many shares get scores close to their real-share ratio.
     - Users with few shares are pulled toward 0.5 (neutral).
   - Parameters:
     - confidence_threshold controls how quickly confidence saturates.

4. Feature engineering (per news item)
   - Aggregate trust scores of sharers:
     - avg_user_trust
     - min_user_trust
     - max_user_trust
     - trust_std
     - num_shares
     - high_trust_ratio (proportion above threshold, e.g., > 0.7)
   - Label: 1 for fake, 0 for real.

5. Modeling
   - Classifier: Small feedforward neural network for tabular features.
   - Input size: 6 features (as listed above).
   - Train/validation split: Hold out a test set.
   - Optimization: Cross-entropy loss with Adam.

6. Evaluation
   - Metrics: Accuracy; optionally precision, recall, F1.
   - Analysis ideas: Confusion matrix; trust score distribution; feature distributions by class.

## Project Structure

- data/: Raw CSVs (politifact_fake.csv, politifact_real.csv).
- src/:
  - data_preprocessing.py
  - graph_construction.py
  - trustworthiness.py
  - features.py
  - model.py
  - train_eval.py
- notebooks/: Exploratory analysis and visualizations.
- reports/: Results, figures, and final write-up.
- README.md: This document.

## Library Requirements

- Python ecosystem:
  - pandas
  - numpy
  - networkx
  - scikit-learn
  - torch
  - matplotlib
  - seaborn

Installation (example):
- pip install networkx pandas numpy torch scikit-learn matplotlib seaborn

## Assumptions and Notes

- Users are represented via tweet IDs or user IDs derived from sharing posts.
- A user’s trustworthiness is undefined only when they have zero shares; default to neutral (0.5).
- Confidence threshold tuning affects calibration of trust scores; start around 5 and adjust.
- Feature normalization (e.g., StandardScaler) is recommended before training.

## Risks and Mitigations

- Class imbalance: Use stratified splits; consider class weighting or resampling.
- Data leakage: Ensure that user-based statistics are computed only on training data when doing strict evaluations; for a class project, a simple split may suffice, but document the choice.
- Sparsity: Many items may have few sharers; consider minimum-share filtering or smoothing.
- Overfitting: Use dropout, early stopping, and hold-out validation.

## Extension Ideas

- Add temporal features (e.g., early sharer trust, diffusion speed).
- Community detection on the user subgraph; include community-level trust features.
- Incorporate text features from titles or content using embeddings.
- Try graph-based models (e.g., GNNs on the bipartite graph).
- Calibrate outputs (Platt scaling or isotonic regression) for better probabilities.

## Deliverables

1. Network visualization highlighting user communities and sharing patterns.
2. Trustworthiness distribution plots (histogram, KDE).
3. Model performance: accuracy, precision, recall, F1, and confusion matrix.
4. Feature importance or sensitivity analysis (e.g., permutation importance on the tabular model).
5. Clean, documented code repository with clear instructions to run each step.

## Suggested Timeline

- Step 1: Data preprocessing, network construction.
- Step 2: Trustworthiness computation, feature engineering.
- Step 3: Model implementation and training; initial evaluations.
- Step 4: Final evaluation, visualizations, report writing and polishing.

## How to Run (High-Level)

- Prepare data in data/ with expected CSV fields.
- Execute preprocessing to create user–news pairs.
- Build the bipartite graph and compute user statistics.
- Compute trustworthiness scores and extract per-news features.
- Normalize features, split data, train the model, and evaluate.
- Generate plots and compile findings into the report.

## Ethical Considerations

- Avoid labeling individuals as malicious; trust scores reflect observed sharing behavior under dataset constraints.
- Be transparent about limitations and dataset biases.
- Do not deploy beyond the classroom context without rigorous validation and appropriate safeguards.

## Acknowledgments

- Inspired by misinformation detection research leveraging social propagation and user credibility signals.
- Dataset attribution to PolitiFact-style public datasets where applicable.
