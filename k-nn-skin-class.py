import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

# read csv files
skin_dataset = pd.read_csv('skin-dataset.csv')
non_skin_dataset = pd.read_csv('nonskin-dataset.csv')

def plot_dataset(r, g, b, k):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot the initial pandas dataset
    ax.scatter(skin_dataset["R"].tolist(), skin_dataset["G"].tolist(), skin_dataset["B"].tolist(), c = 'b', marker='o', label='Skin Samples')
    ax.scatter(non_skin_dataset["R"].tolist(), non_skin_dataset["G"].tolist(), non_skin_dataset["B"].tolist(), c = 'r', marker='o', label='Non-Skin Samples')
    # plot the input value on the graph
    ax.scatter(r, g, b, c='g', marker='o', label='Input Value')

    # get joint dataframe with distances
    distances_df = calc_distances(skin_dataset, non_skin_dataset, r, g, b)
    # plot the nearest neighbours on the dataframe
    plot_k_neighbours(r, g, b, k, distances_df, ax)

    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')
    plt.title(str(k) + " - Nearest Neighbour")
    plt.legend(loc='best')

    plt.show()

# calculate distance using the manhattan equation
def manhat_distance_xyz(x1, y1, z1, x2, y2, z2):
    return abs((x1 - x2)) + abs((y1 - y2)) + abs((z1 - z2))

# return a combined datatable with distances
def calc_distances(skin_dataset, non_skin_dataset, r, g, b):
    skin_dataset['Type'] = 'Skin'
    non_skin_dataset['Type'] = 'Non-Skin'

    combined_df = pd.concat([skin_dataset, non_skin_dataset])
    distances = []

    # shouldn't really iterate through a df, but its only small so its fine.
    for _, row in combined_df.iterrows():
        distances.append(manhat_distance_xyz(r, g, b, row['R'], row['G'], row['B']))

    combined_df['Distance'] = distances
    return combined_df

def plot_k_neighbours(r,g,b, k, combined_dataframe, ax):

    count = 0
    for _, row in combined_dataframe.sort_values(by='Distance').iterrows():
        count += 1
        if count > k: break
        # probably a better way to do this, but it works for now.
        if count == 1:
            ax.plot([r, row['R']], [g, row['G']], [b, row['B']], c='grey', linestyle=':', label='Closest Neighbours')
        else:
            ax.plot([r, row['R']], [g, row['G']], [b, row['B']], c='grey', linestyle=':')
        

if __name__ == "__main__":

    r = int(sys.argv[1])
    g = int(sys.argv[2])
    b = int(sys.argv[3])
    k = int(sys.argv[4])

    if r > 255 or g > 255 or b > 255:
        raise Exception("RGB cannot have values greater than 255.")

    if k % 2 == 0:
        print("Consider an odd number for k to increase how effectiveness of the classifier")

    plot_dataset(r, g, b, k)