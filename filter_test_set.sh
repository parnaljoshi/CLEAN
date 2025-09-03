#!/bin/bash

#!/bin/bash

# Copy/paste this job script into a text file and submit with the command:
#    sbatch thefilename
# job standard output will go to the file slurm-%j.out (where %j is the job ID)

#SBATCH --time=24:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=36   # 36 processor core(s) per node 
#SBATCH --mem=300G   # maximum memory per node
#SBATCH --job-name="filtertestset"
#SBATCH --mail-user=parnal@iastate.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

module purge
module load cdhit

# Set input and output filenames
TRAIN_CSV="train_set.csv"
TEST_CSV="test_set.csv"
TRAIN_FASTA="train_for_filter.fasta"
TEST_FASTA="test_for_filter.fasta"
FILTERED_FASTA="filtered_test.fasta"
FILTERED_CSV="filtered_test_set.csv"

# Step 1: Convert train/test CSVs to FASTA
echo "Converting CSV files to FASTA..."
python3 - <<EOF
import pandas as pd

def write_fasta(csv_file, fasta_file):
    df = pd.read_csv(csv_file, sep='\t')
    with open(fasta_file, 'w') as f:
        for _, row in df.iterrows():
            f.write(f">{row['Entry']}\n{row['Sequence']}\n")

write_fasta("$TRAIN_CSV", "$TRAIN_FASTA")
write_fasta("$TEST_CSV", "$TEST_FASTA")
EOF

# Step 2: Run CD-HIT-2D (filters test.fasta against train.fasta at 50% identity)
echo "Running CD-HIT-2D..."
cd-hit-2d -i "$TRAIN_FASTA" -i2 "$TEST_FASTA" -o "$FILTERED_FASTA" -c 0.5 -n 3

# Step 3: Convert filtered FASTA back to CSV
echo "Converting filtered FASTA back to CSV..."
python3 - <<EOF
import pandas as pd
from Bio import SeqIO

fasta_file = "$FILTERED_FASTA"
original_csv = "$TEST_CSV"
output_csv = "$FILTERED_CSV"

original_df = pd.read_csv(original_csv, sep='\t')
keep_ids = {record.id for record in SeqIO.parse(fasta_file, "fasta")}
filtered_df = original_df[original_df['Entry'].isin(keep_ids)]
filtered_df.to_csv(output_csv, sep='\t', index=False)
EOF

echo "Filtered test set saved to: $FILTERED_CSV"
