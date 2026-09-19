"""
Purpose:    Summarize AlphaGenome variant-effect scores for rs11174281 (chr12:39917691 C>T)
            to support an MR analysis treating the variant as an LCU instrument. We separate
            RNA-seq tissue ranking (brain-focused), per-gene expression direction, and
            brain regulatory-element overlap so each Mendelian-randomization-relevant claim is
            traceable to a specific track. raw_score for RNA_SEQ/CAGE = log2(ALT/REF) for the
            T (minor) allele, so its sign gives direction directly.
Inputs:     rs11174281_allscorers_scores.csv  (all-modality tidy scores, 22150 tracks)
            rs11174281_rnaseq_variant_scores.csv (RNA_SEQ GeneMaskLFC scores per gene/tissue)
Outputs:    Console tables (RNA-seq tissue ranking, brain ranking, top genes, brain reg overlap)
            rs11174281_rnaseq_tissue_ranking.csv
            rs11174281_top_genes.csv
            rs11174281_brain_regulatory_overlap.csv
Key Steps:  Load -> tag brain tissues -> rank RNA-seq tracks by |raw| -> per-gene direction
            -> filter DNASE/ATAC/histone tracks to brain biosamples for regulatory overlap.
How to Run: python3 analyze_rs11174281.py
"""
import pandas as pd
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)

D = "/Users/youzhiwei/Downloads/nyu_data_task/task4_ai_agents/output/plots_pathA/"
allsc = pd.read_csv(D + "rs11174281_allscorers_scores.csv")
rna_g = pd.read_csv(D + "rs11174281_rnaseq_variant_scores.csv")

BRAIN = ("brain","cortex","cortical","hippocamp","striat","caudate","putamen","nucleus accumbens",
         "cerebell","neuron","cerebral","frontal","amygdala","substantia nigra","spinal","glia",
         "astrocyte","oligodendro","prefrontal","tibial nerve","basal ganglia")
def is_brain(s):
    s = str(s).lower()
    return any(k in s for k in BRAIN)

# ---------- Item 3: RNA-seq tissue ranking (brain focus) ----------
rna = allsc[allsc.output_type == "RNA_SEQ"].copy()
rna["abs"] = rna.raw_score.abs()
rna["label"] = rna.biosample_name.fillna(rna.gtex_tissue).fillna(rna.track_name)
rna["brain"] = (rna.label.map(is_brain) | rna.gtex_tissue.map(is_brain))
print("RNA_SEQ tracks:", len(rna), "| brain tracks:", int(rna.brain.sum()))

top_tissues = (rna.sort_values("abs", ascending=False)
               .loc[:, ["gene_name","label","gtex_tissue","brain","raw_score","quantile_score"]].head(20))
print("\n=== TOP 20 RNA-seq tracks by |effect| (all tissues) ===")
print(top_tissues.to_string(index=False))

brain_rna = rna[rna.brain].sort_values("abs", ascending=False)
print("\n=== TOP 15 BRAIN RNA-seq tracks by |effect| ===")
print(brain_rna[["gene_name","label","gtex_tissue","raw_score","quantile_score"]].head(15).to_string(index=False))
top_tissues.to_csv(D + "rs11174281_rnaseq_tissue_ranking.csv", index=False)

# ---------- Item 4: genes with largest expression change + direction ----------
# Aggregate per gene across tracks: take the track of max |raw| (most affected context).
g = (rna.sort_values("abs", ascending=False).groupby("gene_name", as_index=False).first())
g = g.sort_values("abs", ascending=False)
g["direction_for_T(alt)"] = g.raw_score.apply(lambda x: "UP" if x > 0 else "DOWN")
print("\n=== TOP 12 GENES by max |RNA-seq effect| (direction = for ALT=T allele) ===")
print(g[["gene_name","label","raw_score","direction_for_T(alt)","quantile_score"]].head(12).to_string(index=False))
g[["gene_name","label","raw_score","direction_for_T(alt)","quantile_score","brain"]].head(20).to_csv(
    D + "rs11174281_top_genes.csv", index=False)

# also the GeneMaskLFC per-gene scorer summary (mean over tracks)
print("\n=== GeneMaskLFC per-gene (mean raw over its tracks) ===")
gm = (rna_g.groupby("gene_name", as_index=False)
      .agg(mean_raw=("raw_score","mean"), max_abs=("raw_score", lambda s: s.abs().max()), n=("raw_score","size")))
gm = gm.reindex(gm.max_abs.sort_values(ascending=False).index)
print(gm.head(10).to_string(index=False))

# ---------- Item 5: brain regulatory-element overlap (DNase / ATAC / histone) ----------
reg = allsc[allsc.output_type.isin(["DNASE","ATAC","CHIP_HISTONE"])].copy()
reg["abs"] = reg.raw_score.abs()
reg["label"] = reg.biosample_name.fillna(reg.track_name)
reg["brain"] = reg.label.map(is_brain)
breg = reg[reg.brain].sort_values("abs", ascending=False)
print("\n=== BRAIN regulatory tracks (DNase/ATAC/histone), top 20 by |effect| ===")
cols = ["output_type","label","histone_mark","raw_score","quantile_score"]
print(breg[cols].head(20).to_string(index=False))
print("\nbrain regulatory track counts by modality:\n", breg.output_type.value_counts().to_string())
breg[cols + ["ontology_curie"]].to_csv(D + "rs11174281_brain_regulatory_overlap.csv", index=False)
