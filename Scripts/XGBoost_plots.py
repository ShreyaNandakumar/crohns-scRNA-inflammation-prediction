import numpy as np
import matplotlib.pyplot as plt
import os

#STEP 1: load feature importance data and average across folds

#make output directory for the plots
os.makedirs("Results/XGBoost/new_plots", exist_ok=True)

base_dirs = {"best_auroc": "Results/XGBoost/AUROC_run",
             "best_balanced": "Results/XGBoost/BALANCED_run"}

#loop over both feature importance combos and make plots for each
for combo_name, folder in base_dirs.items():

    #load importances from all 5 folds and average them
    all_importances = []

    for fold in range(5):
        
        if combo_name == "best_auroc":
            folder = "Results/XGBoost/AUROC_run"
        else:
            folder = "Results/XGBoost/BALANCED_run"

        imp_path = f"{folder}/feature_importance_{combo_name}_fold{fold}.csv"
        
        #read the csv manually 
        importances = []

        with open(imp_path, "r") as f:
            lines = f.readlines()
            for line in lines[1:]: #skip header
                parts = line.strip().split(",")
                importances.append(float(parts[1])) #importance is in second column

        all_importances.append(importances)

    #average importance across all 5 folds
    avg_importances = np.mean(all_importances, axis=0)
    std_importances = np.std(all_importances, axis=0)

    #PC labels
    n_pcs = len(avg_importances)
    pc_labels = ["PC" + str(i + 1) for i in range(n_pcs)]

    #STEP 2: make bar plot of average importances with error bars for std
    fig, ax = plt.subplots(figsize=(10, 5))

    #bar chart with error bars showing standard deviation across folds
    ax.bar(pc_labels, avg_importances, yerr=std_importances, capsize=5, color="skyblue", edgecolor="black")
    ax.set_title("XGBoost Feature Importance (" + combo_name + ")\naveraged across 5 folds")
    ax.set_xlabel("Principal Components")
    ax.set_ylabel("Average Importance (Gain)")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()

    #save the plots
    plot_path = "Results/XGBoost/new_plots/feature_importance_" + combo_name + ".png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print("Plot saved to: ", plot_path)

print("\n ALL PATHS DONEEEEE!!")