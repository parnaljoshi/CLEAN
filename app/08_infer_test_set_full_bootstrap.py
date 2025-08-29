import pandas as pd
import numpy as np
from CLEAN.infer import infer_maxsep
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import os

def run_bootstrap_inference(train_data, test_data_file, model_name,
                            n_bootstrap=1000, pretrained=False, output_dir="./bootstrap_temp", custom_test_csv=False):
    os.makedirs(output_dir, exist_ok=True)

    test_df = pd.read_csv(f"./data/{test_data_file}.csv")

    metrics = {
        'precision': [], 'recall': [], 'f1': [], 'accuracy': []
    }

    rng = np.random.RandomState(42)

    for i in range(n_bootstrap):
        # Sample test proteins with replacement
        # sampled_df = test_df.sample(n=len(test_df), replace=True, random_state=rng)
        sampled_df = test_df.sample(n=len(test_df), replace=True, random_state=rng.randint(0, 100000))

        # Save bootstrap replicate
        replicate_file = os.path.join(output_dir, f"test_replicate_all_{i}.csv")
        print(replicate_file)
        sampled_df.to_csv(replicate_file, index=False)

        # Run inference on this replicate
        y_true, y_pred = infer_maxsep(
                train_data=train_data,
                test_data=f"{replicate_file}",
                report_metrics=False,
                pretrained=pretrained,
                model_name=model_name,
                return_preds=True,
                custom_test_csv=True  # <-- You must modify infer_maxsep to support this
            )
        
        # Flatten predictions if needed
        y_true = [x[0] if isinstance(x, (list, tuple)) else x for x in y_true]
        y_pred = [x[0] if isinstance(x, (list, tuple)) else x for x in y_pred]

        # Compute metrics
        metrics['precision'].append(precision_score(y_true, y_pred, average='macro', zero_division=0))
        metrics['recall'].append(recall_score(y_true, y_pred, average='macro', zero_division=0))
        metrics['f1'].append(f1_score(y_true, y_pred, average='macro', zero_division=0))
        metrics['accuracy'].append(accuracy_score(y_true, y_pred))

    # Compute CIs
    ci = {
        metric: (np.mean(values), np.percentile(values, 2.5), np.percentile(values, 97.5))
        for metric, values in metrics.items()
    }

    return ci

# Run the bootstrap inference
ci_results = run_bootstrap_inference(
    train_data="train_set",
    test_data_file="filtered_test_set",
    model_name="train_set_all_triplet_6000_batch",
    pretrained=False,
    n_bootstrap=1000,  # Use 1000 for final runs
)

# Print results
print("All EC Original")
for metric, (mean, lower, upper) in ci_results.items():
    print(f"{metric.capitalize()}: {mean:.3f}  CI: [{lower:.3f}, {upper:.3f}]")