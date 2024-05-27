with open('maize/maize.pos.5mC.txt', 'r') as f:
    lines = f.readlines()
    # Split the file contents into lines

    # Extract sequences (5th column in each line)
    sequences = [line.split('\t')[4] for line in lines[:50000]]

    # Join the sequences with a newline character to prepare them for writing to a file
    sequences_text = '\n'.join(sequences)

    # Writing to a file (the file path will be /mnt/data/sequences.txt in this case)
    output_file_path = 'maize/pos.txt'
    with open(output_file_path, 'w') as file:
        file.write(sequences_text)

