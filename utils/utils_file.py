def load_from_tsv_data(path, skip_head=False):
    sequences = []
    labels = []
    dic = {'A':1, 'T': 2, 'C': 3, 'G': 4}
    with open(path, 'r') as file:
        if skip_head:
            next(file)
        for line in file:
            if line[-1] == '\n':
                line = line[:-1]
            l = line.split('\t')
            assert len(l[0]) == 41
            assert len(l[1]) == 1
            sequences.append([])
            sequences[-1].append(0)
            for i in l[0]:
                sequences[-1].append(dic[i])
                # 0: mc4, 1:hmc5, 2:ma6, 3:mc5, 4:neg
            if l[1] == '1':
                labels.append(0)
            elif l[1] == '2':
                labels.append(1)
            elif l[1] == '3':
                labels.append(2)
            elif l[1] == '4':
                labels.append(3)
            elif l[1] == '5':
                labels.append(4)
            else:
                print('the label is wrong')
    return sequences, labels
