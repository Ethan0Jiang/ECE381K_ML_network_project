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
- xxx/:
  - xxxx

## Library Requirements

- Python ecosystem:
  - pandas
  - numpy
  - networkx
  - scikit-learn
  - torch
  - matplotlib
  - seaborn


## Deliverables

1. Network visualization.
2. Trustworthiness distribution plots (histogram, KDE).
3. Model performance: accuracy, precision, recall, F1, and confusion matrix.
5. Clean, documented code repository with clear instructions to run each step.

## Suggested Timeline

- Step 1: Data preprocessing, network construction.
- Step 2: Trustworthiness computation, feature engineering.
- Step 3: Model implementation and training; initial evaluations.
- Step 4: Final evaluation, visualizations, report writing and polishing.


