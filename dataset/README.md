# Dataset Information

This dataset contains a subset of data from CLFW

The dataset collection procedure according to the following configuration:

- Include the latest version of all pages of domains that have been labeled as `Cybercrime`.
- Exclude pages of domains that have been labeled as `Child Sexual Abuse Material (CSAM)`. Be aware that this does not guarantee that this data does not contain CSAM as some pages might have been mislabeled.
- There was no duplication removal strategy selected, meaning we did not take any measures to remove pages with duplicate data.

# Files

- html: a folder containing all snapshots that meet the requirements
- metadata.csv: a file containing the meta data of the downloaded html files

### One way to load the meta data

```python
import pandas as pd
df = pd.read_csv('metadata.csv', on_bad_lines="skip", engine="python")
```
