"""
Script to train personalized models for all patients in the dataset
"""

import argparse
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from src.data_loader import D1NAMODataLoader
from src.pipeline import run_personalized_training


def train_all_patients(data_dir, output_dir, task='hypoglycemia', threshold=70):
    """
    Train personalized models for all patients
    
    Args:
        data_dir: Directory containing patient data
        output_dir: Directory to save models and results
        task: 'hypoglycemia' or 'hyperglycemia'
        threshold: Glucose threshold (70 for hypo, 180 for hyper)
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all patient IDs
    patient_ids = [p.name for p in data_dir.iterdir() if p.is_dir()]
    
    if len(patient_ids) == 0:
        print(f"No patient directories found in {data_dir}")
        print("Tip: For testing, use synthetic data by running example.py")
        return
    
    print(f"Found {len(patient_ids)} patients")
    print(f"Task: {task} detection (threshold: {threshold} mg/dL)")
    print(f"Output directory: {output_dir}")
    print()
    
    # Initialize data loader
    loader = D1NAMODataLoader(data_dir=data_dir)
    
    # Store results
    all_results = {}
    failed_patients = []
    
    # Train models for each patient
    for patient_id in tqdm(patient_ids, desc="Training models"):
        try:
            print(f"\nProcessing patient {patient_id}...")
            
            # Load patient data
            patient_data = loader.load_patient_data(patient_id)
            
            if patient_data['ecg_signal'] is None:
                print(f"  Warning: No ECG data for patient {patient_id}")
                failed_patients.append((patient_id, "No ECG data"))
                continue
            
            # Train model
            pipeline, metrics = run_personalized_training(
                patient_data,
                task=task,
                threshold=threshold
            )
            
            if metrics is None:
                print(f"  Warning: Training failed for patient {patient_id}")
                failed_patients.append((patient_id, "Training failed"))
                continue
            
            # Save model (if available)
            if 'MBeat' in pipeline.models:
                model_filename = f"{patient_id}_{task}_thresh{threshold}.pkl"
                model_path = output_dir / 'models' / model_filename
                model_path.parent.mkdir(parents=True, exist_ok=True)
                pipeline.save_model('MBeat', str(model_path))
            else:
                print(f"  Warning: MBeat model not found in pipeline for {patient_id}")
            
            # Store metrics
            all_results[patient_id] = metrics
            
            # Print metrics
            print(f"  Results for {patient_id}:")
            print(f"    AUC: {metrics['auc']:.4f}")
            print(f"    Sensitivity: {metrics['sensitivity']:.4f}")
            print(f"    Specificity: {metrics['specificity']:.4f}")
            print(f"    F1-Score: {metrics['f1_score']:.4f}")
            
        except Exception as e:
            print(f"  Error processing patient {patient_id}: {str(e)}")
            failed_patients.append((patient_id, str(e)))
            continue
    
    # Save results
    if len(all_results) > 0:
        # Save as JSON
        results_file = output_dir / f'results_{task}_thresh{threshold}.json'
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        # Save as CSV
        results_df = pd.DataFrame(all_results).T
        csv_file = output_dir / f'results_{task}_thresh{threshold}.csv'
        results_df.to_csv(csv_file)
        
        # Print summary
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"\nSuccessfully trained models for {len(all_results)} patients")
        print(f"Failed for {len(failed_patients)} patients")
        
        if len(failed_patients) > 0:
            print("\nFailed patients:")
            for patient_id, reason in failed_patients:
                print(f"  - {patient_id}: {reason}")
        
        print("\nAverage Performance Across All Patients:")
        print(results_df.mean().to_string())
        
        print("\nPerformance Standard Deviation:")
        print(results_df.std().to_string())
        
        print(f"\nResults saved to:")
        print(f"  - {results_file}")
        print(f"  - {csv_file}")
        print(f"\nModels saved to: {output_dir / 'models'}")
        print("="*60)
    else:
        print("\nNo models were successfully trained.")


def main():
    parser = argparse.ArgumentParser(
        description='Train personalized glucose monitoring models for all patients'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/raw',
        help='Directory containing patient data (default: data/raw)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/processed',
        help='Directory to save models and results (default: data/processed)'
    )
    parser.add_argument(
        '--task',
        type=str,
        choices=['hypoglycemia', 'hyperglycemia'],
        default='hypoglycemia',
        help='Detection task (default: hypoglycemia)'
    )
    parser.add_argument(
        '--threshold',
        type=int,
        default=None,
        help='Glucose threshold in mg/dL (default: 70 for hypo, 180 for hyper)'
    )
    
    args = parser.parse_args()
    
    # Set default threshold based on task
    if args.threshold is None:
        args.threshold = 70 if args.task == 'hypoglycemia' else 180
    
    # Run training
    train_all_patients(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        task=args.task,
        threshold=args.threshold
    )


if __name__ == '__main__':
    main()
