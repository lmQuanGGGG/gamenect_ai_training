import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_comparison():
    import json
    import os
    
    json_file = "/Users/wang04/Downloads/GAMENECT/gamenect_ai_training/reports/cv_results.json"
    
    models = []
    aucs = []
    f1s = []
    accs = []
    
    if not os.path.exists(json_file):
        print("Không tìm thấy file cv_results.json. Vui lòng chạy lại script train_real_data.py để tạo data trước.")
        return
        
    with open(json_file, "r", encoding="utf-8") as f:
        results = json.load(f)
        
    for name, res in results.items():
        models.append(name)
        aucs.append(res['roc_auc'][0])
        f1s.append(res['f1'][0])
        accs.append(res['accuracy'][0])
        
    if not models:
        print("File cv_results.json trống hoặc không hợp lệ.")
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
