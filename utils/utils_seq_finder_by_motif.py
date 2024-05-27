import random


def find_seq_by_motif(input_path, label, motif, s, e, mask_inside=True, mask_count=1):
    sequences = []
    labels = []
    dic = {'A': 1, 'T': 2, 'C': 3, 'G': 4}

    with (open(input_path, 'r') as file):
        for line in file:
            if not line.startswith('>') and line.strip():
                line = line.strip()  # Removes the newline and any trailing whitespace
                if line[s: e + 1] == motif:
                    # Initialize sequence with '0' as a prefix
                    seq = [0] + [dic[n] for n in line]
                    # Determine the positions available for masking
                    positions = list(range(s+1, 21)) + list(range(22, e+2)) if mask_inside else list(range(1, s+1)) + list(range(e + 2, len(seq)))
                    # Randomly select positions to mask
                    mask_positions = random.sample(positions, min(mask_count, len(positions)))
                    # Apply mask
                    for pos in mask_positions:
                        seq[pos] = 0  # Mask with '0'
                    # Append the processed sequence and label
                    sequences.append(seq)
                    labels.append(label)

    return sequences, labels