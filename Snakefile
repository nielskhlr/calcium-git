# --------------------------------------------------------------
# Calcium‑Imaging‑Workflow (AVI → Motion‑Correction → ROI‑Detection → Analysis)
# --------------------------------------------------------------

# Snakemake automation for the analysis of calcium imaging data.
# Specifically, this workflow performs motion correction and ROI detection on .avi video files.
# ROIs will be determined unsing the potassium stimulation recordings.
# Analysis will be performed on the spontaneous activity recordings, using the ROIs determined from the potassium stimulation recordings.

import glob, os, yaml, re

# --------------------------------------------------------------
# Config file
# --------------------------------------------------------------

configfile: "config.yaml"

# --------------------------------------------------------------
# Finding data files and defining Sample-IDs
# --------------------------------------------------------------

# Determine data files and their basenames for use in the Snakemake workflow.
# First, all potassium stimulation videos are identified.
POTASSIUM = glob.glob(os.path.join(config["raw_dir"], "*_K+_*.avi"))

# Next, the basenames are removed, leaving the sample ID to be used as identifiers for the motion correction and ROI detection steps.
def split_sample_rep(path):
    name = os.path.basename(path)

    m = re.match(r"(.+?)_K\+_(\d+)\.avi$", name)
    if m:
        return m.group(1), m.group(2)

    m = re.match(r"(.+?)_spontan_(\d+)\.avi$", name)
    if m:
        return m.group(1), m.group(2)

    raise ValueError(f"Unexpected format: {name}")

SAMPLES = sorted([split_sample_rep(f) for f in POTASSIUM])

# --------------------------------------------------------------
# Final Outputs
# --------------------------------------------------------------

rule all:
    input:
        expand(os.path.join(config["roi_dir"],"{sample}_{rep}_rois.zip"), sample=[s for s, r in SAMPLES],
            rep=[r for s, r in SAMPLES])

# --------------------------------------------------------------
# Motion Correction and Alignment
# --------------------------------------------------------------

# The motion correction step takes the raw .avi files as input and produces motion-corrected and aligned .avi files as output.
rule motion_correction:
    input:
        k=os.path.join(config["raw_dir"], "{sample}_K+_{rep}.avi"),
        spont=os.path.join(config["raw_dir"], "{sample}_spontan_{rep}.avi")
    output:
        k_corr=os.path.join(config["motioncorrect_dir"], "{sample}_K+_{rep}.avi"),
        spont_aligned=os.path.join(config["motioncorrect_dir"], "{sample}_spontan_{rep}_aligned.avi")
    conda:
        "envs/caiman_environment.yml"
    threads: 4 # Adjust the number of threads based on your system's capabilities
    resources:
        mem_mb = 6000 # Adjust memory requirements based on the expected usage of the motion correction step
    shell:
        """
        export TMPDIR=$(dirname {output.k_corr})
        python scripts/caiman_analysis.py {input.k} {input.spont} 5 {output.k_corr} {output.spont_aligned}
        """

# --------------------------------------------------------------
# ROI Detection
# --------------------------------------------------------------

# The ROI detection step takes the motion-corrected potassium stimulation videos as input and produces .zip files containing the detected 
# ROIs as output.
rule roi_detection:
    input:
        k_corr=os.path.join(config["motioncorrect_dir"], "{sample}_K+_{rep}.avi")
    output:
        zip = os.path.join(config["roi_dir"], "{sample}_{rep}_rois.zip")
    conda:
        "envs/roi_environment.yml"
    threads: 2 # Adjust the number of threads based on your system's capabilities
    shell:
        "python scripts/roiadjust.py {input.k_corr} {output.zip}"