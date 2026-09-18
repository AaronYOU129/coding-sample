# AlphaGenome analysis of rs11174281 — MR instrument for lifetime cannabis use (LCU)

**Variant:** rs11174281 · **GRCh38/hg38 chr12:39,917,691 · C>T** · intronic in *SLC2A13*
**Allele tested:** REF = C → ALT = **T** (the common minor allele; see note on risk allele below)
**Tools:** locally-built `alphagenome_local` MCP (Paper2Agent / google-deepmind/alphagenome), 1 Mb prediction window centred on the variant.

> **Headline:** AlphaGenome predicts rs11174281 is a **non-coding regulatory variant** sitting in active brain chromatin (DNase + H3K4me1/H3K27ac enhancer marks across cortex, hippocampus, striatum, substantia nigra). Its predicted expression effects are **directionally consistent and in the genome-wide extreme tail (quantile ≈ 0.97–0.99) but small in absolute magnitude (max log2FC ≈ 0.009, i.e. <1 %)**. The best-supported targets are **KIF21A (up)** and the lncRNA **LINC02555 (down)**, not the nearby *LRRK2* coding gene.

---

## 1. Variant lookup (chromosome, position, alleles, build)

Resolved via the **Ensembl REST API** (`/variation/human/rs11174281`) and cross-checked against dbSNP:

| Field | Value |
|---|---|
| Build | **GRCh38 (hg38)** — confirmed (`assembly_name: GRCh38`) |
| Chromosome | **chr12** |
| Position | **39,917,691** (1-based) |
| Alleles | **C / A / T** (REF = C; ancestral = C) |
| Minor allele | **T**, global MAF ≈ **0.145** |
| Class / consequence | SNP / **intron_variant** |
| Host gene | ***SLC2A13*** (ENSG00000151229, minus strand, chr12:39,755,025–40,106,094) |
| Synonyms | rs61619773, rs58193704 |

The position falls deep within an intron of *SLC2A13*; the broader 1 Mb neighbourhood also contains *KIF21A* (centromeric), *LRRK2* / *LRRK2-DT* / *MUC19* (telomeric), and several lncRNAs (*LINC02555*, *LINC02471*).

> **Note on the LCU-risk allele.** AlphaGenome reports effects of ALT relative to REF; it does not know the GWAS effect direction. I tested **C→T** (T = minor allele, the conventional effect allele). **All directions below are stated for the T allele.** If the within-family LCU GWAS reports **T** as the risk/increasing allele, the directions stand as written; if **C** is the risk allele, flip every sign. The rarer A allele was not modelled. Please confirm the coded/risk allele from your GWAS summary stats before importing directions into the MR.

---

## 2. Variant effect prediction across all modalities

Ran `alphagenome_score_variant` (all recommended scorers, 1 Mb window) → **22,150 track scores** saved to `rs11174281_allscorers_scores.csv`. Modality coverage:

| Modality | Scorer | # tracks | Interpretation |
|---|---|---|---|
| RNA_SEQ | GeneMaskLFCScorer | 13,860 | gene expression (log2 fold-change) |
| CHIP_TF | CenterMask (501 bp) | 3,234 | TF binding |
| CHIP_HISTONE | CenterMask (2001 bp) | 2,232 | histone marks |
| CAGE | CenterMask (501 bp) | 1,092 | TSS activity |
| DNASE | CenterMask (501 bp) | 610 | chromatin accessibility |
| SPLICE_SITE_USAGE / SPLICE_JUNCTIONS | splicing scorers | 367 each | splicing |
| ATAC | CenterMask (501 bp) | 334 | chromatin accessibility |
| CONTACT_MAPS | ContactMapScorer | 28 | 3D contacts |
| PROCAP | CenterMask (501 bp) | 24 | nascent transcription |
| SPLICE_SITES | GeneMaskSplicing | 2 | splice sites |

**Modality-level read-out:**
- **Expression (RNA-seq/CAGE):** small but highly-ranked changes (details in §3–4).
- **Chromatin (DNase/ATAC/histone):** the variant overlaps active regulatory chromatin and the strongest brain effects are here (§5).
- **Splicing:** **negligible** — max |effect| was SPLICE_JUNCTIONS 0.061, SPLICE_SITES/USAGE 0.008. AlphaGenome does **not** flag this as a splice-altering variant despite its intronic location.
- **3D contacts (Hi-C):** essentially flat (max 0.001).

> **Data-integrity note.** In the combined all-scorer CSV, the RNA_SEQ `raw_score` column contains duplicated rows where some entries are inflated (e.g. a spurious *LRRK2* monocyte value of 1.98 alongside the correct 0.0007). The **dedicated** `GeneMaskLFCScorer` output (`rs11174281_rnaseq_variant_scores.csv`, range ±0.009, matching the tool's own reported top-genes) is the trustworthy source, so **all RNA-seq magnitudes and directions in §3–4 come from the dedicated scorer**, not the combined CSV.

---

## 3. RNA-seq tissue ranking (brain focus)

From the dedicated RNA_SEQ scorer (6,732 gene×track rows; 578 brain tracks). Effects are reported as **log2 fold-change for the T allele**; `quantile_score` is the percentile vs AlphaGenome's genome-wide background (sign = direction).

**Top RNA-seq tracks overall (all tissues):**

| Gene | Tissue / cell | log2FC (T) | quantile |
|---|---|---|---|
| KIF21A | Purkinje cell (cerebellum) | **+0.0092** | 0.994 |
| ENSG00000199571 (snoRNA) | CD8⁺ T cell | −0.0089 | −0.992 |
| LINC02555 | adrenal gland | −0.0075 | −0.987 |
| LINC02555 | liver / lung / PBMC | ≈ −0.0070 | ≈ −0.985 |
| ENSG00000199571 | CD14⁺ monocyte | −0.0067 | −0.986 |

**Top BRAIN RNA-seq tracks (the MR-relevant subset):**

| Gene | Brain region | log2FC (T) | quantile | Direction (T) |
|---|---|---|---|---|
| **KIF21A** | cerebellar hemisphere | **+0.0060** | 0.980 | UP |
| KIF21A | cerebellum | +0.0056 | 0.976 | UP |
| **LINC02555** | amygdala | −0.0057 | −0.978 | DOWN |
| LINC02555 | caudate nucleus (striatum) | −0.0056 | −0.977 | DOWN |
| LINC02555 | hypothalamus | −0.0056 | −0.977 | DOWN |
| LINC02555 | anterior cingulate cortex | −0.0055 | −0.977 | DOWN |
| LINC02555 | frontal cortex | −0.0054 | −0.976 | DOWN |
| LINC02555 | dorsolateral prefrontal cortex | −0.0053 | −0.974 | DOWN |
| LINC02555 | putamen / substantia nigra | ≈ −0.0053 | −0.972 | DOWN |
| LINC02555 | hippocampus (Ammon's horn) | −0.0051 | −0.970 | DOWN |
| LINC02555 | nucleus accumbens | −0.0050 | −0.968 | DOWN |
| LINC02471 | dorsolateral prefrontal cortex | −0.0050 | −0.984 | DOWN |

**Interpretation:** within brain, the variant produces (i) a **cerebellum-biased up-regulation of *KIF21A*** and (ii) a **remarkably uniform down-regulation of the lncRNA *LINC02555* across every interrogated brain region** — cortex, striatum (caudate/putamen/accumbens), hippocampus, amygdala, hypothalamus, substantia nigra and spinal cord. The striatal/cortical/hippocampal hits are exactly the circuits relevant to a neurobehavioural trait like LCU. Absolute magnitudes are sub-1 %, but the consistency and high quantile rank are notable.

(Ranking tables saved to `rs11174281_rnaseq_tissue_ranking.csv` and `rs11174281_rnaseq_brain_ranking.csv`.)

---

## 4. Gene(s) with largest expression changes + direction (for the T allele)

Per-gene summary from the dedicated scorer (max |log2FC| across tissues, and net mean across brain tracks):

| Gene | Type | Max |log2FC| (tissue) | Direction (T) | Brain mean log2FC | Notes |
|---|---|---|---|---|---|---|
| **KIF21A** | protein-coding | 0.0092 (Purkinje cell) | **UP** | **+0.0025** | strongest single effect; cerebellum-biased |
| **LINC02555** | lncRNA | 0.0075 (adrenal) | **DOWN** | **−0.0038** | most consistent brain effect (all regions) |
| LRRK2-DT | lncRNA (LRRK2 divergent) | 0.0064 (skin) | DOWN | −0.0033 | antisense/divergent to *LRRK2* |
| ENSG00000199571 | snoRNA | 0.0089 (T cell) | DOWN | — | mostly immune, not brain |
| LINC02471 | lncRNA | 0.0050 (DLPFC) | DOWN | −0.0023 | cortical |
| **SLC2A13** | protein-coding (host) | 0.0042 (trophoblast) | DOWN | −0.0008 | host gene; only a weak effect |
| ABCD2 | protein-coding | 0.0048 (monocyte) | DOWN | ≈ 0 in brain | cerebellar Purkinje signal |
| **LRRK2** | protein-coding | **0.0028** | ~neutral | −0.0003 | **negligible — not a predicted target** |

**Largest expression change:** **KIF21A — up-regulated for the T allele** (peak in cerebellar Purkinje cells; consistently up across brain). The largest *down* effect and the most spatially consistent brain signal is **LINC02555 — down-regulated for the T allele in every brain region**.

**Important:** despite the locus's proximity to *LRRK2* (a high-profile gene), AlphaGenome does **not** predict a meaningful change in *LRRK2* coding expression (max |log2FC| ≈ 0.003, brain mean ≈ 0). The only *LRRK2*-associated signal is in its divergent lncRNA *LRRK2-DT* (down). The host gene *SLC2A13* shows only a weak down-regulation. (Per-gene table: `rs11174281_top_genes.csv`.)

---

## 5. Overlap with predicted regulatory elements in brain

Yes — the variant **overlaps active regulatory chromatin in brain**. Ranked by background-normalised quantile (`rs11174281_brain_regulatory_overlap.csv`):

**Chromatin accessibility (DNase/ATAC), brain:**

| Element | Brain context | quantile | Direction (T) |
|---|---|---|---|
| DNase | glutamatergic neuron | −0.96 | reduced accessibility |
| DNase | cerebellar cortex | +0.93 | increased |
| DNase | cerebellum | +0.93 | increased |
| DNase | neuronal stem cell | −0.90 | reduced |
| DNase | dorsolateral prefrontal cortex | −0.89 | reduced |
| DNase | posterior cingulate gyrus | −0.89 | reduced |

**Enhancer / active marks (ChIP-histone), brain:**

| Mark (type) | Brain region | quantile |
|---|---|---|
| **H3K4me1** (enhancer) | spinal cord / DLPFC / hippocampus / cingulate / substantia nigra / caudate | 0.88–0.97 |
| **H3K27ac** (active enhancer) | DLPFC / cingulate / caudate / hippocampus / substantia nigra | 0.84–0.87 |
| H3K4me3 (promoter) | neuron | 0.87 |

**Interpretation:** the SNP lies in a region carrying **DNase accessibility plus H3K4me1 + H3K27ac** — the canonical signature of an **active enhancer** — across precisely the brain regions implicated in addiction/reward (DLPFC, hippocampus, caudate, substantia nigra, cingulate). The T allele is predicted to **modulate** (mostly reduce, except cerebellum) accessibility/enhancer activity here. This supports a **cis-regulatory / enhancer mechanism** rather than a coding or splicing effect.

---

## 6. Regulatory-landscape plots saved

All PNGs are REF (grey) vs ALT=T (red) overlays centred on chr12:39,917,691 (orange line):

1. **`rs11174281_brain_chromatin_variant_effects.png`** — the regulatory landscape: DNase (hippocampus/cerebellum/DLPFC) + multiple histone marks (H3K27ac, H3K4me1, H3K4me3, H3K36me3, H3K9me3) across caudate nucleus, cerebellum, substantia nigra and DLPFC, with gene track. Shows accessible/enhancer chromatin spanning the variant; REF/ALT differences are subtle, consistent with the small predicted effect sizes.
2. **`rs11174281_brain_reglandscape_zoom_variant_effects.png`** — ~43 kb zoom, brain RNA-seq tracks (frontal cortex, cerebellum, DLPFC) within the *SLC2A13* intron.
3. **`rs11174281_gene_context_wide_variant_effects.png`** — ~500 kb context showing *REDIC1* and *SLC2A13* gene bodies and brain RNA-seq, locating the variant within *SLC2A13*.

---

## 7. Summary — AlphaGenome's best guess at the gene & mechanism

**Mechanism:** rs11174281 (chr12:39,917,691 C>T, GRCh38) is an **intronic non-coding variant in *SLC2A13* that behaves like a brain cis-regulatory / enhancer variant.** It sits in chromatin marked by DNase accessibility and H3K4me1 + H3K27ac (active-enhancer signature) across reward/cognition-relevant brain regions (DLPFC, hippocampus, caudate/striatum, substantia nigra, cingulate, cerebellum). AlphaGenome predicts **no meaningful splicing effect and no meaningful change in *LRRK2* coding expression**, arguing against the "obvious" *LRRK2* candidate.

**Best-guess target gene(s):**
- **KIF21A — up-regulated by the T allele**, the single largest predicted expression effect, cerebellum-biased but present across brain.
- **LINC02555 — down-regulated by the T allele**, the most *consistent* brain signal (uniform across cortex, striatum, hippocampus, amygdala, hypothalamus, substantia nigra).
- Secondary/weaker: *LRRK2-DT* (down), *LINC02471* (down, cortical), host *SLC2A13* (weak down).

**For the MR:** AlphaGenome supports rs11174281 acting through **altered brain enhancer activity that re-tunes local gene expression (chiefly KIF21A up / LINC02555 down) in addiction-relevant circuits**, rather than via coding, splicing, or *LRRK2*. This is **mechanistically plausible biological support** for a brain-expressed cis-regulatory instrument — but two caveats matter for instrument validity:
1. **Effect sizes are small in absolute terms** (max log2FC ≈ 0.009, <1 %), though in the genome-wide extreme tail (quantile ≈ 0.97–0.99) — typical of a common non-coding GWAS hit.
2. **Multiple genes respond** (KIF21A, LINC02555, LRRK2-DT, SLC2A13). This is a **potential horizontal-pleiotropy flag**: the instrument may influence several transcripts, so a single clean gene→LCU→outcome path should not be assumed without colocalisation / multi-gene sensitivity analysis.
3. **Confirm the coded/risk allele** in your within-family GWAS; all directions here are for ALT = T and must be flipped if C is the LCU-increasing allele.

---

## Files saved under `output/plots_pathA/`

| File | Contents |
|---|---|
| `rs11174281_allscorers_scores.csv` | All 22,150 track scores, every modality (item 2) |
| `rs11174281_rnaseq_variant_scores.csv` | Dedicated RNA_SEQ GeneMaskLFC scores (authoritative; items 3–4) |
| `rs11174281_rnaseq_tissue_ranking.csv` | Top RNA-seq tracks, all tissues (item 3) |
| `rs11174281_rnaseq_brain_ranking.csv` | Top brain RNA-seq tracks (item 3) |
| `rs11174281_top_genes.csv` | Per-gene max effect + direction for T (item 4) |
| `rs11174281_brain_regulatory_overlap.csv` | Brain DNase/ATAC/histone overlap (item 5) |
| `rs11174281_brain_chromatin_variant_effects.png` | Brain DNase + histone regulatory landscape (item 6) |
| `rs11174281_brain_reglandscape_zoom_variant_effects.png` | Zoomed brain RNA-seq at variant (item 6) |
| `rs11174281_gene_context_wide_variant_effects.png` | Wide gene-context view (item 6) |
| `analyze_rs11174281.py` | Analysis script (reproducible) |

*Build: GRCh38/hg38. All numbers derive from actual AlphaGenome MCP tool calls; the rarer A allele was not modelled; RNA-seq values use the dedicated GeneMaskLFC scorer due to artifacts in the combined CSV's RNA_SEQ column.*
