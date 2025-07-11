import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
'''
    Plot a subplot of a particular type of metric during training, e.g. val_acc / loss / acc.

     Parameters:
        position: int position of the subplot, e.g. 131, 132, 133
        histories: dict optimiser name → history dictionary
        key: str the field to plot, e.g. ‘acc’, ‘val_acc’, ‘loss’
        title: str Title of the subplot
        ylabel: str y-axis name
        baseline: float optional baseline level (e.g. accuracy of untrained model)
'''


def plot_metric_subplot(position, histories, key, title, ylabel, baseline=None):

    plt.subplot(position)
    colors = ['#f1c40f','#9b59b6', '#1abc9c','#95a5a6','#d35400','#27ae60']
    if baseline is not None:
        plt.axhline(y=baseline, color='gray', linestyle='--', label='Untrained')
    for idx, (name, history) in enumerate(histories.items()):
        if key in history:
            this_color = colors[idx % len(colors)]
            plt.plot(history[key], label=name, color=this_color)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)

def plot_all_histories(histories, initial={'acc':None, 'loss':None, 'f1':None,'val_acc':None,'val_loss':None,'val_f1':None}):
    plt.figure(figsize=(18, 8))
    plot_metric_subplot(231, histories, 'val_acc', "Validation Accuracy", "Val Accuracy",baseline=initial['val_acc'])
    plot_metric_subplot(232, histories, 'acc', "Training Accuracy", "Train Accuracy",baseline=initial['acc'])
    plot_metric_subplot(233, histories, 'val_loss', "Validation Loss", "Val Loss", baseline=initial['val_loss'])
    plot_metric_subplot(234, histories, 'loss', "Training Loss", "Loss",baseline=initial['loss'])
    plot_metric_subplot(235, histories, 'val_f1', "Validation Micro F1 Scores", "Val F1 Scores", baseline=initial['val_f1'])
    plot_metric_subplot(236, histories, 'f1', "Training Micro F1 Scores", "F1 Scores", baseline=initial['f1'])
    plt.tight_layout()
    plt.show()


def plot_feature_distributions(original_data, standardized_data, feature_indices=[0,1,2], color='#6495ED'):
    """
    Visualising the distribution of raw and normalised data across several feature dimensions
    """
    feature_names = [f"Feature_{i + 1}" for i in feature_indices]


    for i, idx in enumerate(feature_indices):
        plt.figure(figsize=(12, 5))

        feature_vals = original_data[:, idx]
        x_range = (np.min(feature_vals) - 1, np.max(feature_vals) + 1)

        # Raw data distribution
        plt.subplot(1, 2, 1)
        sns.histplot(original_data[:, idx], kde=True, color=color,bins=30)
        plt.title(f"Original Feature {idx + 1}")
        plt.xlabel(f"Feature {idx + 1}")
        plt.xlim(x_range)

        # Standardised data distribution
        plt.subplot(1, 2, 2)
        sns.histplot(standardized_data[:, idx], kde=True, color=color,bins=30)
        plt.title(f"Standardized Feature {idx + 1}")
        plt.xlabel(f"Feature {idx + 1}")
        plt.xlim(x_range)

        plt.tight_layout()
        #plt.savefig(f"feature_{idx + 1}_distribution.png", dpi=300)
        plt.show()

'''
  Generate heat maps for training and validation sets based on metrics results

    Parameters:
        metrics: dict, key in the format ‘lr=0.01_bs=64’, value in the metrics dictionary
        k1,k2: write characters in simplified form(e.g. ‘lr=0.01_bs=64’ indicates k1 =lr, k2 =bs)
        value_key: str, name of the metric to be extracted (e.g. ‘Micro F1 Scores’)
        title_prefix: str, for chart title
'''

def plot_heatmap_from_metrics(metrics,k1,k2,value_key='F1', title_prefix='Micro F1 Scores',x_label='Learning Rate',y_label ="Batch Size",sw_float=True):
    # Extract the unique learning rate and batch size
    print('metrics', metrics)
    p1 = sorted(list(set(float(k.split('_')[0].split('=')[1]) for k in metrics)))
    if sw_float:
        p2 = sorted(list(set(float(k.split('_')[1].split('=')[1]) for k in metrics)))
    else:
        p2 = sorted(list(set(int(k.split('_')[1].split('=')[1]) for k in metrics)))
    # Initialise the score matrix
    train_matrix = np.zeros((len(p2), len(p1)))
    val_matrix = np.zeros((len(p2), len(p1)))

    for i, v2 in enumerate(p2):
        for j, v1 in enumerate(p1):
            key = f'{k1}={v1}_{k2}={v2}'
            train_score = metrics[key].get(f'Train {value_key}', np.nan)
            val_score = metrics[key].get(f'Val {value_key}', np.nan)
            train_matrix[i, j] = train_score
            val_matrix[i, j] = val_score

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(train_matrix, xticklabels=p1, yticklabels=p2, annot=True, fmt=".4f", cmap="RdBu_r",
                ax=axs[0])
    axs[0].set_title(f'Train {title_prefix} Score Heatmap')
    axs[0].set_xlabel(x_label)
    axs[0].set_ylabel(y_label)

    sns.heatmap(val_matrix, xticklabels=p1, yticklabels=p2, annot=True, fmt=".4f", cmap="RdBu_r",
                ax=axs[1])
    axs[1].set_title(f'Validation {title_prefix} Score Heatmap')
    axs[1].set_xlabel(x_label)
    axs[1].set_ylabel(y_label)

    plt.tight_layout()
    plt.show()
