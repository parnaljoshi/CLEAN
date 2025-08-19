from Bio import SeqIO
import csv

# Load ec_uniprot_all.tsv into a dictionary
ec_map = {}
with open("ec_uniprot_all.tsv") as f:
    next(f)  # Skip header
    for line in f:
        ec, uniprot = line.strip().split()
        ec_map[uniprot] = ec

# Parse modified_headers.fasta
output_rows = []
for record in SeqIO.parse("modified_headers.fasta", "fasta"):
    header_parts = record.description.split('|')
    if len(header_parts) >= 2:
        uniprot_id = header_parts[1]
        ec_number = ec_map.get(uniprot_id)
        if ec_number:
            output_rows.append([uniprot_id, ec_number, str(record.seq)])

# Write to output file
with open("combined_output_for_CLEAN.tsv", "w", newline="") as out_f:
    writer = csv.writer(out_f, delimiter='\t')
    writer.writerow(["Entry", "EC number", "Sequence"])
    writer.writerows(output_rows)

