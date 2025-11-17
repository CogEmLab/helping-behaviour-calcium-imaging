# Helping Behavior Analysis Pipeline

## Batch CNMF-E Processing

This part of the pipeline processes miniscope `.tiff` files using **CNMF-E**. It performs motion correction, CNMF-E initialization and refinement, component evaluation, and saving of all outputs (models, dF/F, QC plots, session metadata).

All processing parameters are defined in YAML configuration files.

### Running the Batch Pipeline

Run the full pipeline using the main configuration:

```bash
python batch_cnmfe.py --config config/master.yaml
```

This will:

- Search all folders listed in `data_paths`
- Locate TIFF files under `*/miniscope/*.tiff`
- Run motion correction + CNMF-E
- Save models, PNR & correlation images, dF/F, and QC plots
- Update the session CSV file

### Using Configuration Files

#### Main config

```
config/master.yaml
```

Contains all default parameters:
- Paths to datasets
- Paths to session metadata files
- CNMF-E parameters
- Motion-correction settings
- Column requirements
- Output locations

IMPORTANT: You normally do **not** edit this file.

### Override Configs

Override configs let you modify only a few parameters without touching `master.yaml`.

Run with one or more overrides:

```bash
python batch_cnmfe.py     --config config/master.yaml     --override config/override_synthetic.yaml
```

Multiple overrides can be chained:

```bash
python batch_cnmfe.py     --override config/debug_fast.yaml config/test_subset.yaml
```

Overrides always take priority over the main config.

### EXAMPLE 1: Running CNMF-E on a Small Subset of Data

Create e.g. `config/test_subset.yaml`:

```yaml
caiman:
  general:
    data_paths:
      - "data/Cohort01/"
    session_file_paths:
      - "SESSION_FILE_subset.csv"
```

Run:

```bash
python batch_cnmfe.py --override config/test_subset.yaml
```

### EXAMPLE 2: Running CNMF-E With Faster Debug Parameters

Example override: `config/debug_fast.yaml`

```yaml
caiman:
  cnmfe:
    only_init: true
    tsub: 4
    ssub: 2
  general:
    skip_if_exists: false
```

Run:

```bash
python batch_cnmfe.py --override config/debug_fast.yaml
```

This setup allows you to:
- Run the full dataset  
- Test rapidly on small subsets  
- Use synthetic datasets for validation  
- Change parameters safely via overrides  

All without modifying the main pipeline script.
