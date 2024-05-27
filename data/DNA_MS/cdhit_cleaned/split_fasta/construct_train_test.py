import os
from pathlib import Path
import random


def process_fasta_to_tsv(base_dir, max_sequences, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize a dictionary to store all sequences
    all_sequences = {'train': [], 'test': []}
    record = {}
    label = {'4mC': 1, '5hmC': 2, '6mA': 3, '5mC': 4, '6mA-neg': 5}
    # Loop through each modification and species
    for modification_dir in base_dir.glob('*/'):
        modification = modification_dir.name
        for species_dir in modification_dir.glob('*/'):
            species = species_dir.name
            sequences = []
            if modification == '6mA-neg':
                f_name = '*combined_neg.fasta'
            else:
                f_name = '*combined_pos.fasta'
            # Collect all sequences from '_pos.fasta' files
            for fasta_file in species_dir.glob(f_name):
                with open(fasta_file, 'r') as f:
                    for line in f:
                        if not line.startswith('>') and line.strip():
                            sequences.append(line.strip())

            # Shuffle and take a subset if necessary
            random.shuffle(sequences)
            if modification == '6mA' or modification == '6mA-neg':
                if len(sequences) > max_sequences-7000:
                    sequences = sequences[:max_sequences-7000]

            else:
                if len(sequences) > max_sequences:
                    sequences = sequences[:max_sequences]

            # Split into training and testing
            split_point = int(len(sequences) * 0.875)
            train_seqs = sequences[:split_point]
            test_seqs = sequences[split_point:]

            # Add sequences to the all_sequences dictionary
            all_sequences['train'].extend([(seq, label[modification]) for seq in train_seqs])
            all_sequences['test'].extend([(seq, label[modification]) for seq in test_seqs])

            # Record the counts
            record[(modification, species)] = (len(train_seqs), len(test_seqs))

    # Write the sequences to train and test .tsv files
    for set_type, seqs in all_sequences.items():
        with open(os.path.join(output_dir, f"{set_type}_set.tsv"), 'w') as tsv_file:
            for seq, mod in seqs:
                tsv_file.write(f"{seq}\t{mod}\n")

    # Write the record counts to a txt file
    with open(os.path.join(output_dir, 'record_counts.txt'), 'w') as record_file:
        for key, (train_count, test_count) in record.items():
            mod, species = key
            record_file.write(f"{mod}\t{species}\t{train_count}\t{test_count}\n")


# Define the base directory and output directory
base_dir = Path('D:\project\DNApred_ResNet\data\DNA_MS\cdhit_cleaned\split_fasta')
output_dir = Path('D:\project\DNApred_ResNet\data\DNA_MS\cdhit_cleaned\split_fasta')

# Run the processing function
process_fasta_to_tsv(base_dir, 10000, output_dir)


# Function to append neg sequences to the previously created train and test sets

