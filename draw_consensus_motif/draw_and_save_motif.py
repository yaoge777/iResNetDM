import logomaker
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_logo(score_path, data_path, save_name, save_best=0):

    score = np.loadtxt(score_path, delimiter=',')
    print(score)
    data = np.loadtxt(data_path, delimiter=',')
    columns = ['A', 'C', 'G', 'T']
    df = pd.DataFrame(data.T, columns=columns)
    print(df.to_string(index=False))
    logo = logomaker.Logo(df)

    ax = logo.ax

    ax.axis('off')
    # rect_start_pos = 5.5   # Replace with your desired start position
    # rect_end_pos = 12.5     # Replace with your desired end position
    # rect_height = 1    # Replace with the height of the rectangle you want
    # # The y position is set to 0, the bottom of your plot
    # rect = patches.Rectangle((rect_start_pos, -rect_height), rect_end_pos - rect_start_pos, rect_height*2,
    #                          edgecolor='blue', facecolor='none', linestyle='--', linewidth=2)
    # ax.add_patch(rect)
    if save_best == 1:
        plt.savefig(save_name)
    plt.show()

score_path = '4mC_FV/score.txt'
data_path = '4mC_FV/3_motif.txt'
save_path = '4mC_CE/4mC_CE_1_0.036.png'
save_best = 0
draw_logo(score_path=score_path, data_path=data_path, save_name=save_path, save_best=save_best)

