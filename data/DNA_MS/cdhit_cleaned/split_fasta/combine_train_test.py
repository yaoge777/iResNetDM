import os

# Define a directory where the fasta files are located
# This should be replaced with the actual directory containing your fasta files
fasta_directory = 'D:\project\DNApred_ResNet\data\DNA_MS\cdhit_cleaned\split_fasta'
modification = '6mA'
species = 'Xoc BLS256'
fasta_directory = os.path.join(fasta_directory, modification, species)
# Define output files for combined positive and negative fasta sequences
output_pos_fasta = 'combined_pos.fasta'
output_neg_fasta = 'combined_neg.fasta'

# Initialize content holders for positive and negative sequences
pos_sequences = []
neg_sequences = []

# Iterate over fasta files in the directory and combine them based on their suffix ('neg' or 'pos')
for filename in os.listdir(fasta_directory):
    if filename.endswith('_neg.fasta'):
        with open(os.path.join(fasta_directory, filename), 'r') as file:
            neg_sequences.extend(file.readlines())
    elif filename.endswith('_pos.fasta'):
        with open(os.path.join(fasta_directory, filename), 'r') as file:
            pos_sequences.extend(file.readlines())

# Write combined sequences to their respective output fasta files
with open(os.path.join(fasta_directory, modification + '_' + species + '_' + output_neg_fasta), 'w') as file:
    file.writelines(neg_sequences)

with open(os.path.join(fasta_directory, modification + '_' + species + '_' + output_pos_fasta), 'w') as file:
    file.writelines(pos_sequences)

# This script will create two new fasta files in the fasta directory: one combining all negative sequences
# and one combining all positive sequences.
