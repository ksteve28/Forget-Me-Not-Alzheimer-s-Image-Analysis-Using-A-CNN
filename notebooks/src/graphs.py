

def counts():
    labels = ['None', 'Very Mild', 'Mild', 'Moderate']
    data = np.array([2560, 717, 1792, 52])
    width = 0.35
    fig, ax = plt.subplots(dpi=150)
    ind = np.arange(4)
    ax.bar(ind, data, width, color='cornflowerblue')
    ax.set_xticks(ind)
    ax.set_xticklabels(labels)
    ax.set_title('Dataset Features')
    plt.tight_layout()
    plt.savefig('../images/graph_data.jpg')
    plt.show()