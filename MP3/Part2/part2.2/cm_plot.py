import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb


def cm_plot():
    f = open("4o.txt", 'r')
    lines = f.readlines()
    matrix = []
    for j in range(10):
        line = lines[j].strip()
        line = line.split(sep=',')
        temp = []
        for i in line:
            i = float(i)
            i = round(i, 2)
            temp.append(i)
        matrix.append(temp)
    new_matrix = np.array(matrix)
    # new_matrix = np.array(matrix)
    df_cm = pd.DataFrame(matrix, index=[i for i in [str(x) for x in range(10)]],
                         columns=[i for i in [str(x) for x in range(10)]])
    fig, ax = plt.subplots(figsize=(25, 15))  # Sample figsize in inches
    sb.heatmap(df_cm, annot=True, linewidths=.5, ax=ax)
    # sb.heatmap(df_cm, annot=True)
    plt.savefig("cm_4*4.png")


if __name__ == "__main__":
    cm_plot()
