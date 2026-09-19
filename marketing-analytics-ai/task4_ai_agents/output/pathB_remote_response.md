# AlphaGenome analysis of rs11174281 — MR instrument for lifetime cannabis use (LCU)

**Variant analyzed:** `chr12:39917691 C>T` (GRCh38)
**Tooling:** AlphaGenome remote MCP (`Paper2Agent-alphagenome-mcp.hf.space`), human model, 1 MB sequence context.
**Date:** 2026-06-17

> **Allele/direction convention.** AlphaGenome scores the effect of **ALT relative to REF**, i.e. the effect of the **T allele** (the minor allele) versus the C reference. A positive `raw_score` = the T allele *increases* the signal; negative = T *decreases* it. The within-family LCU GWAS effect/risk allele was not supplied to me. Everything below is reported for the **T allele**; if your GWAS coded **C** (or the rare A allele) as the LCU-risk allele, flip the signs accordingly. See the caveat in item 4.

---

## 1. Variant lookup (chromosome, position, alleles, build)

Resolved with the Ensembl REST API (GRCh38, Ensembl's default assembly) and cross-checked against NCBI dbSNP.

| Field | Value |
|---|---|
| rsID | rs11174281 (merged: rs58193704, rs61619773) |
| **Build** | **GRCh38 / hg38** (confirmed) |
| Chromosome | chr12 |
| Position | 39,917,691 (1-based) |
| Reference allele | **C** (also the ancestral allele) |
| Alternate allele(s) | **A / T** (tri-allelic; T is the common minor allele) |
| Minor allele / MAF | **T**, MAF ≈ 0.145 (gnomAD/1000G/TOPMed) |
| Most severe consequence | intron variant |

- **dbSNP cross-check:** dbSNP build 157 reports the same locus on `NC_000012.12` (GRCh38) at position 39,917,690 (0-based) = 39,917,691 (1-based), REF C — consistent with Ensembl. The earlier GRCh37 coordinate was chr12:38,597,759 (not used here).
- **Allele choice for AlphaGenome:** AlphaGenome takes coordinates, not rsIDs. I scored the common bi-allelic substitution **C>T**. The rare third allele **A** was **not** scored (low frequency; not the typical effect allele). If A is relevant to your instrument, it can be scored separately.
- **Host gene:** the variant lies **inside an intron of `SLC2A13`** (ENSG00000151229, chr12:39,755,025–40,106,094, minus strand) — the only gene overlapping the position.

---

## 2. Variant effect prediction across all modalities

Batch scoring (`score_variants_batch`, 1 MB context) ran **11 scorers**, producing 11,627 track-level scores. Summary by modality (magnitude = max absolute score across all tracks; `quantile_score` is AlphaGenome's genome-wide percentile rank of the effect, ±1 = extreme):

| Modality | # tracks | Max |raw_score| | Max |quantile| | Interpretation |
|---|---|---|---|---|
| **DNASE** (chromatin accessibility) | 305 | **0.271** | **0.976** | Strongest effect — variant sits in/near an accessible element |
| **CHIP_TF** (TF binding) | 1617 | **0.264** | **0.993** | Strong — alters predicted TF (incl. CTCF) occupancy |
| **CHIP_HISTONE** (histone marks) | 1116 | **0.233** | **0.998** | Strong — alters enhancer/active-chromatin marks |
| ATAC | 167 | 0.127 | 0.963 | Moderate accessibility effect |
| PROCAP | 12 | 0.066 | 0.909 | Modest |
| SPLICE_JUNCTIONS | 367 | 0.061 | 0.975 | Modest |
| CAGE | 546 | 0.054 | 0.901 | Modest (promoter/TSS activity) |
| **RNA_SEQ** (expression) | 7128 | **0.009** | **0.994** | Effect is tiny in magnitude but high-percentile |
| SPLICE_SITE_USAGE | 367 | 0.008 | 0.930 | Negligible |
| SPLICE_SITES | 2 | 0.008 | 0.506 | Negligible |

**Headline:** rs11174281 behaves like a **regulatory / chromatin variant**, not a coding or splicing variant. Its largest effects are on **chromatin accessibility (DNase/ATAC), TF binding, and histone marks** (top-1–3% genome-wide percentiles), while its direct effect on steady-state RNA abundance is small in absolute terms — consistent with a non-coding regulatory element of modest effect size, exactly the profile expected for a common GWAS instrument.

---

## 3. RNA-seq tissue ranking (brain focus)

AlphaGenome carries 667 human RNA-seq tracks. Ranking tissues by the maximum absolute RNA-seq effect across all genes in the window:

**Top tissues overall** (note the #1 is a cerebellar cell type):

| Rank | Tissue / cell type | Max |raw_score| | Brain? |
|---|---|---|---|
| 1 | Purkinje cell (cerebellum) | 0.0092 | ✓ (neural) |
| 2 | CD8⁺ T cell | 0.0089 | |
| 3 | CD4⁺ T cell | 0.0087 | |
| 4 | adrenal gland | 0.0075 | |
| 5 | myotube | 0.0071 | |

**Brain regions ranked** (the trait-relevant set):

| Rank | Brain region | Max |raw_score| |
|---|---|---|
| 1 | Cerebellar hemisphere | 0.0060 |
| 2 | Dorsolateral prefrontal cortex | 0.0059 |
| 3 | Amygdala | 0.0057 |
| 4 | Cerebellum | 0.0056 |
| 5 | Caudate nucleus (striatum) | 0.0056 |
| 6 | Hypothalamus | 0.0056 |
| 7 | Anterior cingulate cortex | 0.0055 |
| 8 | Frontal cortex | 0.0054 |
| 9 | Cervical spinal cord (C1) | 0.0053 |
| 10 | Putamen (striatum) | 0.0053 |
| 11 | Substantia nigra | 0.0052 |
| 12 | Nucleus accumbens (striatum) | 0.0050 |
| 13 | Pituitary gland | 0.0048 |
| 14 | Brain (whole) | 0.0048 |
| 15 | Spinal cord | 0.0046 |

**Interpretation:** brain regions — led by **cerebellum** and **prefrontal cortex**, with all major LCU-relevant structures (striatum = caudate/putamen/nucleus accumbens, amygdala, ACC) represented — are among the most responsive tissues. The absolute effect sizes are small everywhere (≤1% expression change), so the *direction* and *consistency* of the signal (item 4) are more informative than the magnitude.

---

## 4. Gene(s) with the largest expression change + direction (for the T allele)

17 genes fall in the 1 Mb window. Ranked by maximum absolute RNA-seq effect:

| Rank | Gene | Ensembl ID | Max |raw_score| | Notes |
|---|---|---|---|---|
| 1 | **KIF21A** | ENSG00000139116 | **0.0092** | ~474 kb upstream of the variant; within window |
| 2 | (RNA gene) ENSG00000199571 | — | 0.0089 | small RNA |
| 3 | **SLC2A13** | ENSG00000151229 | 0.0078 | **host gene** (variant is intronic) |
| 4 | LINC02555 | ENSG00000260943 | 0.0075 | lncRNA |
| 5 | LRRK2-DT | ENSG00000225342 | 0.0064 | LRRK2 divergent transcript |
| 6 | ABCD2 | ENSG00000173208 | 0.0048 | |

**Direction (effect of the T allele), in brain:**

- **KIF21A — consistently UP.** Across **all 24 brain RNA-seq tracks the effect is positive (24 up / 0 down)**, i.e. the **T allele increases KIF21A expression** in every brain region. The effect is strongest and highest-percentile in **cerebellar hemisphere (+0.0060, quantile 0.98)** and **cerebellum (+0.0056, quantile 0.98)**, with the prefrontal cortex, anterior cingulate, and whole brain all positive (quantiles 0.92–0.93). This is the **single most directionally consistent expression signal** for the variant.
- **SLC2A13 (host gene) — mixed/weak.** 28 brain tracks up vs 20 down; smaller percentile scores (≈0.8). Net effect is a slight increase in some cortical/striatal regions (DLPFC +0.0059, nucleus accumbens +0.0046) but it is not directionally clean.

> **LCU-risk-allele caveat (important for your MR).** These directions are for the **T (minor) allele**. To translate into "effect of the LCU-risk allele," confirm which allele your within-family GWAS coded as the risk/effect allele:
> - If **T = LCU-risk allele** → AlphaGenome predicts the risk allele **raises KIF21A** (and modestly raises SLC2A13 in some regions).
> - If **C = LCU-risk allele** → flip: the risk allele **lowers KIF21A**.
>
> AlphaGenome makes no statement about trait direction; it only predicts molecular direction. Pair this with the sign of your GWAS β to get a mechanistically-oriented MR interpretation.

---

## 5. Overlap with predicted regulatory elements in brain

The variant overlaps an **active, brain-relevant regulatory element**. Brain-track scores (effect of the T allele):

**Chromatin accessibility (DNASE) — brain tracks (15):**

| Brain track | raw_score | quantile |
|---|---|---|
| Glutamatergic neuron | **−0.153** | **−0.96** |
| Dorsolateral prefrontal cortex | −0.100 | −0.89 |
| Cerebellar cortex | **+0.094** | **+0.93** |
| Whole brain | −0.067 | −0.86 |
| Cerebellum | +0.064 | +0.93 |
| Head of caudate nucleus | −0.056 | −0.78 |
| Neuronal stem cell | −0.049 | −0.90 |

→ The variant lies in a region of **predicted open chromatin (a DNase/ATAC element) in brain**, and the T allele **changes accessibility**, *decreasing* it in cortical neurons (glutamatergic neuron, DLPFC, whole brain) but *increasing* it in cerebellum — a tissue-specific, opposing-direction effect.

**Enhancer / histone marks (CHIP_HISTONE) — brain tracks (61):** the dominant affected mark is **H3K4me1 (enhancer mark)**, plus **H3K27ac (active-enhancer mark)**:

| Brain track | mark | raw_score | quantile |
|---|---|---|---|
| Spinal cord | H3K4me1 | −0.095 | −0.97 |
| Substantia nigra | H3K4me1 | +0.028 | +0.89 |
| Caudate nucleus | H3K4me1 | +0.028 | +0.88 |
| Dorsolateral prefrontal cortex | H3K4me1 | +0.026 | +0.90 |
| Caudate nucleus | H3K27ac | +0.021 | +0.85 |

→ Consistent with the variant sitting in a **brain enhancer (H3K4me1 / H3K27ac)** whose strength the T allele modulates.

**TF binding (CHIP_TF) — brain tracks:** the top brain hit is **CTCF** (DLPFC +0.019; astrocyte +0.011; whole brain −0.006), alongside neuronal TFs (TOX4, ID3, FOSL1 in glutamatergic neurons). A predicted **CTCF** effect is mechanistically notable because CTCF mediates chromatin looping/insulation — a plausible route for an intronic element to act on a distal gene (see item 7).

**Conclusion (item 5): Yes.** rs11174281 overlaps a predicted **brain regulatory element** — an accessible (DNase/ATAC) **enhancer** marked by **H3K4me1/H3K27ac** with **CTCF** occupancy — and the T allele is predicted to perturb it (top 1–4% genome-wide for these assays).

---

## 6. Regulatory-landscape plots

Saved under `./output/plots/` (REF = grey, ALT/T = red where applicable):

1. **`rs11174281_landscape_wide_effects_20260610_030235.png`** — multi-modal REF-vs-ALT variant-effect view over a 700 kb window (RNA-seq, DNase, ChIP-histone + gene annotation), spanning the variant in `SLC2A13` through `KIF21A` upstream. Six brain ontologies (DLPFC, cerebellum, frontal cortex, caudate, nucleus accumbens, glutamatergic neuron).
2. **`rs11174281_dnase_brain.png`** — DNase/ATAC chromatin-accessibility tracks around the variant in brain tissues, with the variant position highlighted. *(The tool stamps a default title "…for colon tissue"; the tracks shown are the brain ontologies actually requested — glutamatergic neuron, brain, frontal cortex, cerebellum, DLPFC.)*
3. **`rs11174281_histone_brain.png`** — histone-modification (enhancer-mark) landscape in brain (DLPFC, caudate, substantia nigra, spinal cord, whole brain) across a 20 kb window centered on the variant, with TSS annotations.

---

## 7. Summary — AlphaGenome's best guess at gene and mechanism

**Most likely mechanism.** rs11174281 (`chr12:39917691 C>T`, GRCh38) is a common, tri-allelic intronic variant inside **`SLC2A13`** that AlphaGenome predicts to act as a **non-coding regulatory variant**, not a coding or splicing one. Its effects concentrate on **chromatin accessibility (DNase/ATAC), enhancer histone marks (H3K4me1/H3K27ac) and TF binding (notably CTCF)** — all in the top few percent of genome-wide effects — indicating it sits in and perturbs an **active brain enhancer / accessible element**.

**Most likely target gene.** Although the variant is physically inside `SLC2A13`, AlphaGenome's **cleanest and most consistent expression prediction is on `KIF21A`** (~474 kb upstream, within the 1 Mb window): the **T allele up-regulates KIF21A in 24/24 brain tracks**, most strongly in **cerebellum** and **prefrontal cortex**. The host gene `SLC2A13` shows only a weak, mixed-direction response. The predicted **CTCF**-binding change offers a coherent route for long-range action (altered looping/insulation between the intronic element and the KIF21A locus). So AlphaGenome's best guess is a **distal-enhancer mechanism: the variant tweaks a brain enhancer/CTCF site in the SLC2A13 intron and modestly raises KIF21A expression (T allele) across brain, especially cerebellum/cortex**, with possible secondary modulation of SLC2A13 itself.

**Effect-size context for MR.** Magnitudes are small (≤1% expression change; chromatin effects larger but still moderate). This is the expected fingerprint of a **common-variant instrument with a small per-allele molecular effect** — useful as an MR instrument precisely because it is a weak-but-real cis-regulator, not a large-effect coding change.

**Caveats / what to confirm.**
- **Risk-allele orientation:** directions are for the **T (minor) allele**; map to your GWAS effect allele before interpreting MR direction (item 4). The rare **A** allele was not modeled.
- **Predictions, not measurements:** AlphaGenome outputs are model predictions; the KIF21A link is a computational hypothesis and should be triangulated against brain **eQTL/Hi-C** evidence (e.g. GTEx, PsychENCODE) for SLC2A13 vs KIF21A.
- **Gene-vs-host mismatch:** the strongest expression target (KIF21A) differs from the host gene (SLC2A13) — worth explicit follow-up before assigning the MR exposure mechanism to a specific gene.

---

### Plot files saved (`./output/plots/`)
- `rs11174281_landscape_wide_effects_20260610_030235.png` — 700 kb multi-modal REF-vs-ALT effects (RNA-seq, DNase, ChIP-histone, gene annotation), brain ontologies
- `rs11174281_dnase_brain.png` — brain DNase/ATAC accessibility around the variant
- `rs11174281_histone_brain.png` — brain histone-mark (enhancer) landscape around the variant

### Supporting data files (`./output/`)
- `rs11174281_scores.csv` — full batch variant-scoring table (11 scorers, 11,627 track scores)
- `rs11174281.tsv` — input variant table uploaded to the MCP
- `meta_human.csv` — AlphaGenome human output-track metadata (used to select brain ontology terms)
