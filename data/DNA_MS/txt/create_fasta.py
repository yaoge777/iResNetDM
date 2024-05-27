import os
import glob
#
# 设置你的工作目录到modifications的上级目录
os.chdir('D:\project\DNApred_ResNet\data\DNA_MS\\txt')

# 定义一个新的文件夹来存放所有的.fasta文件
os.makedirs('fasta_files', exist_ok=True)

# 遍历modifications文件夹
for modification in os.listdir('.'):
    print(modification)
    if modification != '5mC':
        continue
    for species in os.listdir(modification):
            print(species)
            species_path = os.path.join(modification, species)
            if os.path.isdir(species_path):
                print(True)
                # 遍历每一个species文件夹下的txt文件
                for txt_file in glob.glob(species_path + '/*.txt'):
                    fasta_file = txt_file.replace('.txt', '.fasta')
                    with open(txt_file, 'r') as f_in, open(fasta_file, 'w') as f_out:
                        # 对于每一行sequence，写入一个fasta格式的entry
                        for i, sequence in enumerate(f_in):
                            f_out.write(f'>seq{i}\n{sequence}\n')
                    # 移动fasta文件到新的文件夹中
                    os.rename(fasta_file, os.path.join('fasta_files', species+os.path.basename(fasta_file)))

print('Conversion to FASTA completed.')