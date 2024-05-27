import os


# We will define a function to split a large fasta file into smaller ones based on their headers.
def split_fasta_by_header(large_fasta_path, output_dir):
    """
    Splits a large fasta file into multiple smaller fasta files based on the header prefix.

    :param large_fasta_path: Path to the large input fasta file.
    :param output_dir: Directory where the split fasta files will be saved.
    """
    # Open the large fasta file for reading
    with open(large_fasta_path, 'r') as large_fasta:
        current_file = None
        current_header = None

        for line in large_fasta:
            # Check if the line is a header
            if line.startswith('>'):
                # Extract the file name from the header
                header_prefix = line[1:].split('_seq')[0]

                # If the header has changed, we need to close the old file and open a new one
                if current_header != header_prefix:
                    # Close the previous file if it exists
                    if current_file:
                        current_file.close()

                    # Update the current header
                    current_header = header_prefix

                    # Create a new file for the new header
                    new_file_path = os.path.join(output_dir, f"{header_prefix}.fasta")
                    current_file = open(new_file_path, 'w')

            # Write the line to the current file
            if current_file:
                current_file.write(line)

        # Close the last file
        if current_file:
            current_file.close()


# Replace 'your_large_fasta.fasta' with the path to your large fasta file.
large_fasta_path = '5mC_output.fasta'
# Replace 'output_directory' with the path to the directory where you want the small fasta files.
output_dir = '../split_fasta'

# Make sure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Now call the function with the path to your large fasta file and the output directory
split_fasta_by_header(large_fasta_path, output_dir)