# PCA — two different things

## 1. Semantic PCA (every question)

`app/analytics/semantic_pca.py`

```
query embedding  +  PCA_NEIGHBORS nearest chunk vectors (from Chroma)
        │
        ├── cosine similarity      full vectors, via sklearn
        ├── KMeans clustering      full vectors
        └── PCA(n_components=2)    fitted on the documents, then used to
                                   transform BOTH the documents and the query
```

Why fit once and transform both: the query and the passages must live in the
same projection, or the picture is meaningless.

Why similarity comes from the full vectors: two components typically retain
well under half the variance, so distance on the plot is an illustration. The
numbers people read must come from the real space.

### What it reports

* `query_cluster` — KMeans assignment of the query vector
* `closest_cluster` — a similarity-weighted vote among the top passages
  (a genuinely different question from the one above, and they can disagree)
* `closest` — ranked list with document, page and cosine score
* `nearby_other_cluster` — what else sits near, outside the closest cluster
* `cluster_terms` — words frequent inside a cluster and rare outside it
* `explanation` — Gemini's answer to "why are they close?", written only from
  the retrieved passages

### When it declines

`available=False` with a reason when the index is empty or there are fewer
than four neighbours. The question is still answered.

## 2. Dataset PCA (Data Analysis page)

`app/analytics/pca.py`

```
CSV / Excel → numeric columns → drop empty rows → SimpleImputer(median)
            → drop zero-variance columns → StandardScaler → PCA(2) → KMeans
```

Standardisation is mandatory: PCA follows variance, so an unscaled column in
millions would drown one in percent.

Returns explained variance, cumulative variance, loadings, coordinates and
cluster labels — or `performed=False` and a plain reason (fewer than 2 varying
numeric features, fewer than 5 rows, all-constant columns, non-numeric data).

## They are never mixed

Semantic PCA projects **embedding vectors** (meaning). Dataset PCA projects
**measured numeric features**. Separate modules, separate schemas, separate
pages.
