import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_comparison():
    # Hardcoded or parsed metrics from CV
    # We will put the values here after we know them, or we can just parse the log directly.
    # Let's write a simple parser for the log file
    log_file = "/Users/wang04/.gemini/antigravity-ide/brain/c3276a4a-0b60-4c88-9e57-f1829c25c073/.system_generated/tasks/task-504.log"
    
    models = []
    aucs = []
    f1s = []
    accs = []
    
    with open(log_file, "r") as f:
        lines = f.readlines()
        
    start_parsing = False
    for line in lines:
        if "Model" in line and "AUC" in line and "F1" in line and "Acc" in line:
            start_parsing = True
            continue
        if start_parsing:
            if "─" in line or "=" in line:
                continue
            if line.strip() == "":
                break
            parts = line.split()
            if len(parts) >= 4:
                # "Gradient Boosting              0.9385±0.003  0.7850±0.004  0.8123±0.003"
                try:
                    name = " ".join(parts[:-3])
                    auc = float(parts[-3].split("±")[0])
                    f1 = float(parts[-2].split("±")[0])
                    acc = float(parts[-1].split("±")[0])
                    models.append(name)
                    aucs.append(auc)
                    f1s.append(f1)
                    accs.append(acc)
                except Exception as e:
                    pass
    
    if not models:
        print("Not finished or could not parse models.")
        return

    x = np.arange(len(models))
    width = 0.22

    fig, ax = plt.subplots(figsize=(12, 7))
    rects1 = ax.bar(x - width, aucs, width, label='AUC', color='royalblue')
    rects2 = ax.bar(x, f1s, width, label='F1 Score', color='coral')
    rects3 = ax.bar(x + width, accs, width, label='Accuracy', color='mediumseagreen')

    ax.set_ylabel('Scores', fontsize=12)
    ax.set_title('So Sánh Hiệu Năng Các Thuật Toán Máy Học (GameNect AI)', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(loc='lower right', fontsize=11)
    ax.set_ylim(0, 1.1)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, rotation=0, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    out_path = "/Users/wang04/Downloads/GAMENECT/gamenect_ai_training/reports/model_comparison.png"
    plt.savefig(out_path, dpi=120)
    print(f"Saved comparison plot to {out_path}")

if __name__ == "__main__":
    plot_comparison()
