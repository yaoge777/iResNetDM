import os

# Assuming all fasta files are in the current working directory, you can list all files and filter for '.fasta' extension
fasta_files = [f for f in os.listdir('.') if f.endswith('.fasta')]

# This will be the combined fasta file
combined_fasta_path = '5mC_combined.fasta'

# Write the contents of each fasta file into the combined fasta file
# Prepend the original file name (without the .fasta extension) to each sequence header
with open(combined_fasta_path, 'w') as combined_fasta:
    for fasta_file in fasta_files:
        with open(fasta_file, 'r') as individual_fasta:
            # We'll use the file name without '.fasta' as the sequence identifier
            seq_id = os.path.splitext(fasta_file)[0]
            for line in individual_fasta:
                if line.startswith('>'):
                    # Add the sequence identifier at the beginning of the header line
                    combined_fasta.write(f">{seq_id}_{line[1:]}")
                else:
                    combined_fasta.write(line)

