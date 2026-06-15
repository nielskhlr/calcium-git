# Snakemake automation for the analysis of calcium imaging data.
# Specifically, this workflow performs motion correction and ROI detection on .avi video files.
# ROIs will be determined unsing the potassium stimulation recordings.
# Analysis will be performed on the spontaneous activity recordings, using the ROIs determined from the potassium stimulation recordings.

import glob
import os

# Determine data files and their basenames for use in the Snakemake workflow.
# First, all potassium stimulation videos are identified.
POTASSIUM = glob.glob("data/input/*K+_*.avi")

# Next, the basenames are extracted to be used as identifiers for the motion correction and ROI detection steps.
def get_base(path):
    name = os.path.basename(path)
    name = name.replace("K+_", "").replace("spontan_", "")
    return os.path.splitext(name)[0]

BASENAMES = [get_base(f) for f in POTASSIUM]

rule all:
    input:
        expand("data/roi/{base}.zip", base=BASENAMES)

# The motion correction step takes the raw .avi files as input and produces motion-corrected and aligned .avi files as output.
rule motion_correction:
    input:
        k="data/input/{prefix}K+_{num}.avi",
        spont="data/input/{prefix}spontan_{num}.avi"
    output:
        k="data/motion_corrected/{prefix}K+_{num}.avi",
        spont="data/motion_corrected/{prefix}spontan_{num}.avi"
    conda:
        "envs/caiman_environment.yml"
    shell:
        "python scripts/caiman_analysis.py {input.k} {input.spont} 5 {output.k} {output.spont}"

# The ROI detection step takes the motion-corrected potassium stimulation videos as input and produces .zip files containing the detected 
# ROIs as output.
rule roi_detection:
    input:
        "data/motion_corrected/{prefix}K+_{num}.avi"
    output:
        "data/roi/{prefix}{num}.zip"
    conda:
        "envs/roi_environment.yml"
    shell:
        "python scripts/roiadjustment.py {input} {output}"