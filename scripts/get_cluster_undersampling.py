import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
import os
from pathlib import Path
import json
import traceback

# ==========================================================
# Set global random seed for reproducibility
# ==========================================================
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
import random
random.seed(GLOBAL_SEED)

# ==========================================================
# 1. Load data
# ==========================================================
path_metadata = r"C:\Users\rafae\OneDrive\Documents\PhD_2026\Thesis\Chapter_2_SPbest\SP_best_Repo\data\df_unique_sp_final_classified_with_id_bacillus.csv"
path_features = r"C:\Users\rafae\OneDrive\Documents\PhD_2026\Thesis\Chapter_2_SPbest\SP_best_Repo\data\cosmpad_sp_25.csv"

df_meta = pd.read_csv(path_metadata)
df_feat = pd.read_csv(path_features)

df_meta_subset = df_meta[['id', 'dataset_name', 'label']]
df = df_feat.merge(df_meta_subset, on='id', how='inner')

# ==========================================================
# 2. Create output directories
# ==========================================================
base_output_dir = Path(r"outputs\undersampled")
figures_dir = base_output_dir / "figures"
analysis_dir = base_output_dir / "analysis"
os.makedirs(figures_dir, exist_ok=True)
os.makedirs(analysis_dir, exist_ok=True)

print(f"Figures will be saved to: {figures_dir}")
print(f"Analysis files will be saved to: {analysis_dir}")

# ==========================================================
# 3. Identify datasets with >= 100 samples
# ==========================================================
dataset_sizes = df.groupby('dataset_name').size()
large_datasets = dataset_sizes[dataset_sizes >= 100].index.tolist()

print(f"Found {len(large_datasets)} datasets with >= 100 samples:")
for ds in large_datasets:
    size = dataset_sizes[ds]
    opt_count = df[(df['dataset_name'] == ds) & (df['label'] == 1)].shape[0]
    print(f"  - {ds}: {size} samples ({opt_count} optimal)")

# Initialize list to store all undersampled data
all_undersampled_dfs = []

# ==========================================================
# 4. Process each large dataset
# ==========================================================
feature_cols = [c for c in df.columns if c not in ['id', 'dataset_name', 'label']]
MIN_SAMPLES_PER_CLUSTER = 3

def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {convert_to_serializable(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    return obj

def random_undersampling(df_non, target_n, random_state=GLOBAL_SEED):
    """
    Perform simple random undersampling without clustering.
    Used for comparison against cluster-based method.
    """
    print(f"    Performing random undersampling: {target_n} samples from {len(df_non)}")
    
    if len(df_non) <= target_n:
        return df_non.copy()
    
    result = df_non.sample(n=target_n, random_state=random_state)
    print(f"    Random sampling completed: {len(result)} samples")
    
    return result

def distribute_evenly(df_non, cluster_labels, target_n, random_state=GLOBAL_SEED):
    """
    Distribute samples evenly across clusters when target is too small
    to enforce minimum per cluster.
    """
    df_non = df_non.copy()
    df_non['cluster'] = cluster_labels
    
    n_clusters = len(np.unique(cluster_labels))
    cluster_sizes = df_non.groupby('cluster').size().to_dict()
    
    print(f"    Using even distribution: {target_n} samples across {n_clusters} clusters")
    
    # Calculate how many samples per cluster (as evenly as possible)
    samples_per_cluster = target_n // n_clusters
    remainder = target_n % n_clusters
    
    samples_to_take = {}
    remaining_to_assign = target_n
    
    # First pass: assign base samples
    for cluster in range(n_clusters):
        available = cluster_sizes.get(cluster, 0)
        desired = samples_per_cluster + (1 if cluster < remainder else 0)
        
        take = min(desired, available)
        if take > 0:
            samples_to_take[cluster] = take
            remaining_to_assign -= take
            print(f"    Cluster {cluster}: taking {take}/{available}")
    
    # If we still have samples to assign, give to clusters with remaining capacity
    if remaining_to_assign > 0:
        print(f"    Remaining to distribute: {remaining_to_assign}")
        
        # Find clusters that can take more
        available_clusters = []
        for cluster in range(n_clusters):
            available = cluster_sizes.get(cluster, 0)
            taken = samples_to_take.get(cluster, 0)
            if available > taken:
                available_clusters.append((cluster, available - taken))
        
        # Sort by available capacity
        available_clusters.sort(key=lambda x: x[1], reverse=True)
        
        for cluster, available in available_clusters:
            if remaining_to_assign <= 0:
                break
            take_extra = min(available, remaining_to_assign)
            samples_to_take[cluster] = samples_to_take.get(cluster, 0) + take_extra
            remaining_to_assign -= take_extra
            print(f"    Cluster {cluster}: added extra {take_extra} (now {samples_to_take[cluster]}/{cluster_sizes[cluster]})")
    
    # Sample from each cluster and preserve original indices
    sampled_dfs = []
    total_taken = 0
    
    for cluster, n_take in samples_to_take.items():
        if n_take > 0:
            cluster_data = df_non[df_non['cluster'] == cluster]
            n_actual = min(n_take, len(cluster_data))
            if n_actual > 0:
                sampled = cluster_data.sample(n=n_actual, random_state=random_state)
                sampled_dfs.append(sampled)
                total_taken += n_actual
    
    result = pd.concat(sampled_dfs, ignore_index=False) if sampled_dfs else pd.DataFrame()
    print(f"    Final sampled: {total_taken}/{target_n} samples")
    
    return result

def redistribute_undersampling(df_non, cluster_labels, target_n, min_samples_per_cluster=MIN_SAMPLES_PER_CLUSTER, random_state=GLOBAL_SEED):
    """
    Redistribute sampling quota when some clusters don't have enough samples.
    Ensures at least min_samples_per_cluster from each cluster when possible,
    but adapts when target is too small.
    """
    df_non = df_non.copy()
    df_non['cluster'] = cluster_labels
    
    n_clusters = len(np.unique(cluster_labels))
    cluster_sizes = df_non.groupby('cluster').size().to_dict()
    
    print(f"    Initial target: {target_n} samples across {n_clusters} clusters")
    print(f"    Cluster sizes: {cluster_sizes}")
    print(f"    Minimum samples per cluster: {min_samples_per_cluster}")
    
    # Check if any cluster has less than minimum samples
    small_clusters = [c for c, size in cluster_sizes.items() if size < min_samples_per_cluster]
    if small_clusters:
        print(f"    Warning: Clusters {small_clusters} have less than {min_samples_per_cluster} samples")
    
    # Calculate minimum total samples needed
    min_total_needed = n_clusters * min_samples_per_cluster
    
    # Check if target_n is too small for minimum per cluster
    if target_n < min_total_needed:
        print(f"    Warning: Target ({target_n}) < minimum required ({min_total_needed})")
        print(f"    Will distribute samples evenly without enforcing minimum per cluster")
        return distribute_evenly(df_non, cluster_labels, target_n, random_state)
    
    # Calculate base quota
    samples_per_cluster = target_n // n_clusters
    remainder = target_n % n_clusters
    
    print(f"    Base quota: {samples_per_cluster} per cluster (+{remainder} to distribute)")
    
    # Calculate target based on available samples
    total_available = len(df_non)
    actual_target = min(target_n, total_available)
    
    if actual_target < target_n:
        print(f"    Adjusting target from {target_n} to {actual_target} (only {total_available} available)")
        target_n = actual_target
    
    # Redistribute sampling quota
    samples_to_take = {}
    remaining_to_assign = target_n
    
    # First pass: assign minimum samples to each cluster if possible
    for cluster in range(n_clusters):
        available = cluster_sizes.get(cluster, 0)
        take = min(min_samples_per_cluster, available)
        samples_to_take[cluster] = take
        remaining_to_assign -= take
        print(f"    Cluster {cluster}: assigned minimum {take}/{available}")
    
    if remaining_to_assign > 0:
        print(f"    Remaining to distribute: {remaining_to_assign}")
        
        # Second pass: distribute remaining samples proportionally to cluster size
        available_slots = {}
        for cluster in range(n_clusters):
            available = cluster_sizes.get(cluster, 0)
            taken = samples_to_take.get(cluster, 0)
            if available > taken:
                available_slots[cluster] = available - taken
        
        if available_slots:
            total_available_slots = sum(available_slots.values())
            
            for cluster, slots in available_slots.items():
                proportion = slots / total_available_slots
                extra = int(round(remaining_to_assign * proportion))
                extra = min(extra, slots)
                
                samples_to_take[cluster] = samples_to_take.get(cluster, 0) + extra
                remaining_to_assign -= extra
                print(f"    Cluster {cluster}: added {extra} (now {samples_to_take[cluster]}/{cluster_sizes[cluster]})")
    
    # If still have remaining, assign to largest clusters first
    if remaining_to_assign > 0:
        print(f"    Still {remaining_to_assign} to assign, distributing to largest clusters")
        clusters_by_size = sorted(
            [(c, cluster_sizes.get(c, 0) - samples_to_take.get(c, 0)) 
             for c in range(n_clusters)],
            key=lambda x: x[1], reverse=True
        )
        
        for cluster, available in clusters_by_size:
            if remaining_to_assign <= 0:
                break
            if available > 0:
                take_extra = min(available, remaining_to_assign)
                samples_to_take[cluster] = samples_to_take.get(cluster, 0) + take_extra
                remaining_to_assign -= take_extra
                print(f"    Cluster {cluster}: added final {take_extra} (now {samples_to_take[cluster]}/{cluster_sizes[cluster]})")
    
    # Final check and sampling - preserve original indices
    sampled_dfs = []
    total_taken = 0
    
    for cluster, n_take in samples_to_take.items():
        if n_take > 0:
            cluster_data = df_non[df_non['cluster'] == cluster]
            n_available = len(cluster_data)
            n_actual = min(n_take, n_available)
            
            if n_actual > 0:
                sampled = cluster_data.sample(n=n_actual, random_state=random_state)
                sampled_dfs.append(sampled)
                total_taken += n_actual
                
                if n_actual < n_take:
                    print(f"      Warning: Cluster {cluster} wanted {n_take} but only {n_available} available")
    
    result = pd.concat(sampled_dfs, ignore_index=False) if sampled_dfs else pd.DataFrame()
    print(f"    Final sampled: {total_taken}/{target_n} samples")
    
    return result

# ==========================================================
# Process each dataset
# ==========================================================
for dataset_name in large_datasets:
    try:
        print(f"\n{'='*60}")
        print(f"Processing: {dataset_name}")
        print('='*60)
        
        # Filter dataset
        df_dataset = df[df['dataset_name'] == dataset_name].copy()
        
        # Skip if no optimal samples
        if df_dataset['label'].sum() == 0:
            print(f"  Skipping {dataset_name}: No optimal samples")
            df_dataset['undersampled'] = False
            df_dataset['original_dataset'] = dataset_name
            df_dataset['original_size'] = len(df_dataset)
            df_dataset['undersampled_size'] = len(df_dataset)
            df_dataset['optimal_ratio'] = 0
            df_dataset['selected_K'] = 'original'
            df_dataset['target_nonopt'] = 0
            df_dataset['actual_nonopt'] = 0
            df_dataset['random_undersampled'] = False
            all_undersampled_dfs.append(df_dataset)
            continue
        
        # Prepare data
        X = df_dataset[feature_cols].values
        y = df_dataset['label'].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # UMAP projection (with fixed seed)
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=GLOBAL_SEED)
        X_umap = reducer.fit_transform(X_scaled)
        df_dataset['UMAP1'] = X_umap[:, 0]
        df_dataset['UMAP2'] = X_umap[:, 1]
        
        # Separate optimal and non-optimal
        df_opt = df_dataset[df_dataset['label'] == 1].copy()
        df_non = df_dataset[df_dataset['label'] == 0].copy()
        
        X_non_scaled = X_scaled[df_dataset['label'] == 0]
        num_optimal = len(df_opt)
        N_total_nonopt = 3 * num_optimal  # desired subsample size
        
        print(f"  Optimal: {num_optimal}, Non-optimal available: {len(df_non)}, Target non-opt: {N_total_nonopt}")
        
        # Store analysis results for this dataset
        analysis_results = {
            'dataset_name': dataset_name,
            'original_size': int(len(df_dataset)),
            'optimal_count': int(num_optimal),
            'nonoptimal_count': int(len(df_non)),
            'target_nonoptimal': int(N_total_nonopt),
            'k_analysis': [],
            'random_undersampling_metrics': {}
        }
        
        # Also perform random undersampling for comparison
        random_undersampled = random_undersampling(df_non, N_total_nonopt, random_state=GLOBAL_SEED)
        
        # Compute metrics for random undersampling
        if len(random_undersampled) > 0:
            X_random_umap = random_undersampled[['UMAP1', 'UMAP2']].values
            
            # Coverage area for random
            try:
                if len(X_random_umap) >= 3:
                    hull_random = ConvexHull(X_random_umap)
                    coverage_area_random = float(hull_random.area)
                else:
                    coverage_area_random = 0.0
            except:
                coverage_area_random = 0.0
            
            # Average pairwise distance for random
            if len(X_random_umap) > 1:
                avg_pairwise_dist_random = float(pdist(X_random_umap).mean())
            else:
                avg_pairwise_dist_random = 0.0
            
            analysis_results['random_undersampling_metrics'] = {
                'coverage_area': coverage_area_random,
                'avg_pairwise_dist': avg_pairwise_dist_random,
                'actual_samples': int(len(random_undersampled))
            }
            
            print(f"\n  Random undersampling comparison:")
            print(f"    Coverage area: {coverage_area_random:.3f}")
            print(f"    Avg pairwise dist: {avg_pairwise_dist_random:.3f}")
        
        # If not enough non-optimal samples, keep all
        if len(df_non) <= N_total_nonopt:
            print(f"  Keeping all {len(df_non)} non-optimal samples (target was {N_total_nonopt})")
            balanced_nonopt = df_non.copy()
            best_K = "all"
            
            # Simple visualization for this case
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Non-optimal points
            sns.scatterplot(
                data=df_non,
                x='UMAP1', y='UMAP2',
                color='blue', s=100, alpha=0.4,
                edgecolor='black', linewidth=0.3, label='Non-optimal', ax=ax
            )
            
            # Optimal points
            sns.scatterplot(
                data=df_opt,
                x='UMAP1', y='UMAP2',
                color='red', marker='X', s=200,
                edgecolor='black', linewidth=1, label='Optimal', ax=ax
            )
            
            ax.set_title(f"UMAP Projection - {dataset_name}\n(Keeping all {len(df_non)} non-optimal)", fontsize=14)
            ax.set_xlabel("UMAP1", fontsize=12)
            ax.set_ylabel("UMAP2", fontsize=12)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            # Save plot
            plot_filename = figures_dir / f"{dataset_name.replace('/', '_')}_original.png"
            plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Plot saved: {plot_filename}")
            
            analysis_results['selected_k'] = 'all'
            analysis_results['final_nonoptimal'] = int(len(df_non))
            analysis_results['note'] = 'kept_all_insufficient_nonoptimal'
            
        else:
            # Evaluate K values with redistribution
            k_results = []
            
            # Cap K at number of optimal samples (or 6, whichever is smaller)
            max_allowed_k = min(num_optimal, 6, N_total_nonopt // MIN_SAMPLES_PER_CLUSTER)
            print(f"  Max allowed K (capped at optimal count and min samples): {max_allowed_k}")
            
            for k in range(2, max_allowed_k + 1):
                # Also ensure K is not larger than target samples
                if k > N_total_nonopt:
                    print(f"  Skipping K={k}: K ({k}) > target samples ({N_total_nonopt})")
                    continue
                
                print(f"  Testing K={k}...")
                
                # Cluster non-optimal points (with fixed seed)
                kmeans = KMeans(n_clusters=k, random_state=GLOBAL_SEED, n_init=10)
                cluster_labels = kmeans.fit_predict(X_non_scaled)
                
                # Check cluster sizes
                unique, counts = np.unique(cluster_labels, return_counts=True)
                cluster_size_dict = {int(u): int(c) for u, c in zip(unique, counts)}
                
                if min(counts) < MIN_SAMPLES_PER_CLUSTER:
                    print(f"    Warning: K={k} has clusters with less than {MIN_SAMPLES_PER_CLUSTER} samples")
                    print(f"    Min cluster size: {min(counts)}")
                
                # Undersample with redistribution
                balanced_temp = redistribute_undersampling(
                    df_non, cluster_labels, N_total_nonopt, 
                    min_samples_per_cluster=MIN_SAMPLES_PER_CLUSTER, 
                    random_state=GLOBAL_SEED
                )
                
                if len(balanced_temp) == 0:
                    print(f"    Skipping K={k}: No samples selected")
                    continue
                
                # Compute metrics
                X_sub_umap = balanced_temp[['UMAP1', 'UMAP2']].values
                
                # Coverage area
                try:
                    if len(X_sub_umap) >= 3:
                        hull = ConvexHull(X_sub_umap)
                        coverage_area = float(hull.area)
                    else:
                        coverage_area = 0.0
                except:
                    coverage_area = 0.0
                
                # Average pairwise distance
                if len(X_sub_umap) > 1:
                    avg_pairwise_dist = float(pdist(X_sub_umap).mean())
                else:
                    avg_pairwise_dist = 0.0
                
                # Get cluster labels using the preserved original indices
                sampled_indices = balanced_temp.index
                index_to_cluster = dict(zip(df_non.index, cluster_labels))
                sampled_clusters = [index_to_cluster[idx] for idx in sampled_indices]
                
                balanced_temp = balanced_temp.copy()
                balanced_temp['cluster'] = sampled_clusters
                
                # Print cluster distribution for debugging
                cluster_dist = balanced_temp['cluster'].value_counts().sort_index()
                print(f"    ACTUAL cluster distribution: {cluster_dist.to_dict()}")
                
                # Calculate entropy
                if len(cluster_dist) > 1:
                    cluster_counts = cluster_dist / len(balanced_temp)
                    entropy = float(-(cluster_counts * np.log2(cluster_counts)).sum())
                else:
                    entropy = 0.0
                
                # Silhouette score (on full non-optimal set)
                if k > 1 and len(np.unique(cluster_labels)) > 1:
                    try:
                        sil_score = float(silhouette_score(X_non_scaled, cluster_labels))
                    except:
                        sil_score = None
                else:
                    sil_score = None
                
                # Calculate improvement over random undersampling
                improvement_coverage = ((coverage_area - analysis_results['random_undersampling_metrics']['coverage_area']) / 
                                       analysis_results['random_undersampling_metrics']['coverage_area'] * 100) if analysis_results['random_undersampling_metrics']['coverage_area'] > 0 else 0
                improvement_dist = ((avg_pairwise_dist - analysis_results['random_undersampling_metrics']['avg_pairwise_dist']) / 
                                   analysis_results['random_undersampling_metrics']['avg_pairwise_dist'] * 100) if analysis_results['random_undersampling_metrics']['avg_pairwise_dist'] > 0 else 0
                
                k_results.append({
                    'K': int(k),
                    'coverage_area': coverage_area,
                    'avg_pairwise_dist': avg_pairwise_dist,
                    'cluster_entropy': entropy,
                    'silhouette_score': sil_score,
                    'actual_samples': int(len(balanced_temp)),
                    'cluster_sizes': cluster_size_dict,
                    'min_cluster_size': int(min(counts)),
                    'improvement_over_random_coverage': float(improvement_coverage),
                    'improvement_over_random_dist': float(improvement_dist)
                })
                
                print(f"    Actual samples taken: {len(balanced_temp)}/{N_total_nonopt}")
                print(f"    Metrics - Coverage: {coverage_area:.3f}, Dist: {avg_pairwise_dist:.3f}, Entropy: {entropy:.3f}")
                print(f"    Improvement over random - Coverage: {improvement_coverage:.1f}%, Dist: {improvement_dist:.1f}%")
            
            # Save K analysis results
            analysis_results['k_analysis'] = convert_to_serializable(k_results)
            
            if not k_results:
                print(f"  No valid K values found for {dataset_name}, keeping original")
                balanced_nonopt = df_non.copy()
                best_K = "all"
            else:
                # Filter out any K that's larger than the number of optimal samples
                k_results = [r for r in k_results if r['K'] <= num_optimal]
                
                if not k_results:
                    print(f"  No valid K values found for {dataset_name}, keeping original")
                    balanced_nonopt = df_non.copy()
                    best_K = "all"
                else:
                    # Select best K (now including random comparison in decision)
                    df_results = pd.DataFrame(k_results)
                    
                    # Combined score that favors both coverage and distance
                    df_results['combined_score'] = df_results['coverage_area'] + df_results['avg_pairwise_dist']
                    
                    # Also consider improvement over random
                    df_results['avg_improvement'] = (df_results['improvement_over_random_coverage'] + 
                                                    df_results['improvement_over_random_dist']) / 2
                    
                    # Final score combines absolute performance and improvement over random
                    df_results['sample_efficiency'] = df_results['actual_samples'] / N_total_nonopt
                    df_results['adjusted_score'] = (df_results['combined_score'] * 
                                                   df_results['sample_efficiency'] * 
                                                   (1 + df_results['avg_improvement']/100))
                    
                    best_K_row = df_results.loc[df_results['adjusted_score'].idxmax()]
                    best_K = best_K_row['K']
                    
                    print(f"\n  Selected K={best_K} for final undersampling")
                    print(f"  Results for K={best_K}:")
                    print(f"    Coverage area: {best_K_row['coverage_area']:.3f}")
                    print(f"    Avg pairwise dist: {best_K_row['avg_pairwise_dist']:.3f}")
                    print(f"    Entropy: {best_K_row['cluster_entropy']:.3f}")
                    print(f"    Actual samples: {best_K_row['actual_samples']}/{N_total_nonopt}")
                    print(f"    Improvement over random: {best_K_row['avg_improvement']:.1f}%")
                    
                    # Save K analysis to CSV
                    k_analysis_df = pd.DataFrame(k_results)
                    k_analysis_file = analysis_dir / f"{dataset_name.replace('/', '_')}_k_analysis.csv"
                    k_analysis_df.to_csv(k_analysis_file, index=False)
                    print(f"  K analysis saved to: {k_analysis_file}")
                    
                    # Final undersampling with best K
                    kmeans_best = KMeans(n_clusters=int(best_K), random_state=GLOBAL_SEED, n_init=10)
                    cluster_labels_best = kmeans_best.fit_predict(X_non_scaled)
                    
                    balanced_nonopt = redistribute_undersampling(
                        df_non, cluster_labels_best, N_total_nonopt, 
                        min_samples_per_cluster=MIN_SAMPLES_PER_CLUSTER, 
                        random_state=GLOBAL_SEED
                    )
                    
                    # Add cluster labels using the preserved original indices
                    if len(balanced_nonopt) > 0:
                        sampled_indices = balanced_nonopt.index
                        index_to_cluster = dict(zip(df_non.index, cluster_labels_best))
                        sampled_clusters = [index_to_cluster[idx] for idx in sampled_indices]
                        
                        balanced_nonopt = balanced_nonopt.copy()
                        balanced_nonopt['cluster'] = sampled_clusters
                        
                        # Print final cluster distribution
                        final_dist = balanced_nonopt['cluster'].value_counts().sort_index()
                        print(f"\n  FINAL CLUSTER DISTRIBUTION: {final_dist.to_dict()}")
            
            analysis_results['selected_k'] = str(best_K)
            analysis_results['final_nonoptimal'] = int(len(balanced_nonopt)) if len(balanced_nonopt) > 0 else 0
            
            # Create comparison plot showing both cluster-based and random results
            fig, axes = plt.subplots(1, 2, figsize=(20, 8))
            
            # Plot 1: Cluster-based undersampling
            ax1 = axes[0]
            
            # All non-optimal points (background, transparent)
            sns.scatterplot(
                data=df_non,
                x='UMAP1', y='UMAP2',
                color='lightgray', s=40, alpha=0.2,
                edgecolor='none', ax=ax1
            )
            
            # Selected non-optimal points colored by cluster
            if len(balanced_nonopt) > 0 and 'cluster' in balanced_nonopt.columns:
                markers = ['o', '^', 's', 'D', 'v', '<', '>', 'p', '*', 'h']
                clusters = sorted(balanced_nonopt['cluster'].unique())
                
                for i, cluster in enumerate(clusters):
                    cluster_data = balanced_nonopt[balanced_nonopt['cluster'] == cluster]
                    sns.scatterplot(
                        data=cluster_data,
                        x='UMAP1', y='UMAP2',
                        color=sns.color_palette('tab10')[i % 10],
                        marker=markers[i % len(markers)],
                        s=120, alpha=0.9,
                        edgecolor='black', linewidth=0.8,
                        label=f'Cluster {cluster} (n={len(cluster_data)})',
                        ax=ax1
                    )
            
            # Optimal points
            sns.scatterplot(
                data=df_opt,
                x='UMAP1', y='UMAP2',
                color='red', marker='X', s=200,
                edgecolor='black', linewidth=1.5, label='Optimal', ax=ax1
            )
            
            ax1.set_title(f"Cluster-based Undersampling\n(K={best_K}, Cov={best_K_row['coverage_area']:.2f}, Dist={best_K_row['avg_pairwise_dist']:.2f})", 
                         fontsize=13, fontweight='bold')
            ax1.set_xlabel("UMAP1", fontsize=11)
            ax1.set_ylabel("UMAP2", fontsize=11)
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            
            # Plot 2: Random undersampling
            ax2 = axes[1]
            
            # All non-optimal points (background, transparent)
            sns.scatterplot(
                data=df_non,
                x='UMAP1', y='UMAP2',
                color='lightgray', s=40, alpha=0.2,
                edgecolor='none', ax=ax2
            )
            
            # Randomly selected points
            if len(random_undersampled) > 0:
                sns.scatterplot(
                    data=random_undersampled,
                    x='UMAP1', y='UMAP2',
                    color='blue', s=100, alpha=0.6,
                    edgecolor='black', linewidth=0.5, label='Random Selected', ax=ax2
                )
            
            # Optimal points
            sns.scatterplot(
                data=df_opt,
                x='UMAP1', y='UMAP2',
                color='red', marker='X', s=200,
                edgecolor='black', linewidth=1.5, label='Optimal', ax=ax2
            )
            
            ax2.set_title(f"Random Undersampling\n(Cov={analysis_results['random_undersampling_metrics']['coverage_area']:.2f}, "
                         f"Dist={analysis_results['random_undersampling_metrics']['avg_pairwise_dist']:.2f})", 
                         fontsize=13, fontweight='bold')
            ax2.set_xlabel("UMAP1", fontsize=11)
            ax2.set_ylabel("UMAP2", fontsize=11)
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            
            plt.suptitle(f"Comparison: Cluster-based vs Random Undersampling - {dataset_name}", 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            # Save comparison plot
            comparison_filename = figures_dir / f"{dataset_name.replace('/', '_')}_comparison.png"
            plt.savefig(comparison_filename, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Comparison plot saved to: {comparison_filename}")
        
        # Save analysis results as JSON
        analysis_json = analysis_dir / f"{dataset_name.replace('/', '_')}_analysis.json"
        with open(analysis_json, 'w') as f:
            json.dump(convert_to_serializable(analysis_results), f, indent=2)
        
        # Combine optimal with selected non-optimal
        if len(balanced_nonopt) > 0:
            cols_to_drop = []
            if 'UMAP1' in df_opt.columns:
                cols_to_drop.extend(['UMAP1', 'UMAP2'])
            if 'cluster' in balanced_nonopt.columns:
                cols_to_drop.append('cluster')
            
            df_opt_clean = df_opt.drop(columns=[c for c in cols_to_drop if c in df_opt.columns])
            balanced_nonopt_clean = balanced_nonopt.drop(columns=[c for c in cols_to_drop if c in balanced_nonopt.columns])
            
            df_undersampled = pd.concat([df_opt_clean, balanced_nonopt_clean], ignore_index=True)
        else:
            df_undersampled = df_opt.copy()
            cols_to_drop = ['UMAP1', 'UMAP2']
            df_undersampled = df_undersampled.drop(columns=[c for c in cols_to_drop if c in df_undersampled.columns])
        
        # Add metadata about undersampling
        df_undersampled['undersampled'] = True
        df_undersampled['original_dataset'] = dataset_name
        df_undersampled['original_size'] = len(df_dataset)
        df_undersampled['undersampled_size'] = len(df_undersampled)
        df_undersampled['optimal_ratio'] = num_optimal / len(df_undersampled) if len(df_undersampled) > 0 else 0
        df_undersampled['selected_K'] = str(best_K)
        df_undersampled['target_nonopt'] = N_total_nonopt
        df_undersampled['actual_nonopt'] = len(balanced_nonopt) if len(balanced_nonopt) > 0 else 0
        
        all_undersampled_dfs.append(df_undersampled)
        print(f"  Created undersampled dataset: {len(df_undersampled)} samples ({num_optimal} optimal)")
        
    except Exception as e:
        print(f"  ERROR processing {dataset_name}: {str(e)}")
        traceback.print_exc()
        # Still add the original dataset
        df_dataset = df[df['dataset_name'] == dataset_name].copy()
        df_dataset['undersampled'] = False
        df_dataset['original_dataset'] = dataset_name
        df_dataset['original_size'] = len(df_dataset)
        df_dataset['undersampled_size'] = len(df_dataset)
        df_dataset['optimal_ratio'] = df_dataset['label'].sum() / len(df_dataset) if len(df_dataset) > 0 else 0
        df_dataset['selected_K'] = 'error'
        df_dataset['target_nonopt'] = len(df_dataset[df_dataset['label'] == 0])
        df_dataset['actual_nonopt'] = len(df_dataset[df_dataset['label'] == 0])
        all_undersampled_dfs.append(df_dataset)
        print(f"  Added original dataset due to error: {dataset_name}")
        continue

# ==========================================================
# 5. Combine with datasets that were not undersampled (<100 samples)
# ==========================================================
print(f"\n{'='*60}")
print("Combining all datasets...")
print('='*60)

all_datasets = df['dataset_name'].unique()
small_datasets = [ds for ds in all_datasets if ds not in large_datasets]

for ds in small_datasets:
    df_small = df[df['dataset_name'] == ds].copy()
    df_small['undersampled'] = False
    df_small['original_dataset'] = ds
    df_small['original_size'] = len(df_small)
    df_small['undersampled_size'] = len(df_small)
    df_small['optimal_ratio'] = df_small['label'].sum() / len(df_small) if len(df_small) > 0 else 0
    df_small['selected_K'] = 'original'
    df_small['target_nonopt'] = len(df_small[df_small['label'] == 0])
    df_small['actual_nonopt'] = len(df_small[df_small['label'] == 0])
    all_undersampled_dfs.append(df_small)
    print(f"  Added {ds}: {len(df_small)} samples (kept original)")

# ==========================================================
# 6. Save final undersampled dataset with original columns only
# ==========================================================
final_df = pd.concat(all_undersampled_dfs, ignore_index=True)

# Load the original file to get its exact columns
original_df = pd.read_csv(path_metadata)
original_columns = original_df.columns.tolist()

print(f"Original file has {len(original_columns)} columns: {original_columns}")

# Keep only columns that exist in both original and final_df
cols_to_keep = [col for col in original_columns if col in final_df.columns]
print(f"Keeping {len(cols_to_keep)} columns: {cols_to_keep}")

# Create final dataframe with only original columns
final_df_original_cols = final_df[cols_to_keep].copy()

main_data_dir = Path(r"C:\Users\rafae\OneDrive\Documents\PhD_2026\Thesis\Chapter_2_SPbest\SP_best_Repo\data")
output_csv = main_data_dir / "spbest_undersampled.csv"
final_df_original_cols.to_csv(output_csv, index=False)

print(f"\n{'='*60}")
print(f"Final dataset saved to: {output_csv}")
print(f"Total samples: {len(final_df_original_cols)}")
print(f"Columns in saved file: {final_df_original_cols.columns.tolist()}")
print(f"Total datasets: {final_df_original_cols['dataset_name'].nunique()}")
print('='*60)

# ==========================================================
# 7. Create summary using the original-columns dataframe
# ==========================================================
print("\n" + "="*60)
print("DETAILED SUMMARY BY DATASET")
print("="*60)

summary = []
for ds in sorted(final_df_original_cols['dataset_name'].unique()):
    ds_data = final_df_original_cols[final_df_original_cols['dataset_name'] == ds]
    
    # Get the corresponding entry from the full final_df to access metadata
    ds_full_data = final_df[final_df['dataset_name'] == ds]
    
    original_size = ds_full_data['original_size'].iloc[0] if 'original_size' in ds_full_data.columns else len(ds_data)
    final_size = len(ds_data)
    opt_count = ds_data['label'].sum()
    
    # Get metadata if available
    undersampled = ds_full_data['undersampled'].iloc[0] if 'undersampled' in ds_full_data.columns else False
    selected_k = ds_full_data['selected_K'].iloc[0] if 'selected_K' in ds_full_data.columns else 'original'
    target_nonopt = ds_full_data['target_nonopt'].iloc[0] if 'target_nonopt' in ds_full_data.columns else len(ds_data[ds_data['label'] == 0])
    actual_nonopt = ds_full_data['actual_nonopt'].iloc[0] if 'actual_nonopt' in ds_full_data.columns else len(ds_data[ds_data['label'] == 0])
    
    summary.append({
        'Dataset': ds[:50] + '...' if len(ds) > 50 else ds,
        'Original': original_size,
        'Final': final_size,
        'Optimal': opt_count,
        'Optimal_Ratio': f"{opt_count/final_size:.2f}",
        'Undersampled': 'Yes' if undersampled else 'No',
        'K': selected_k,
        'Target_Non': target_nonopt,
        'Actual_Non': actual_nonopt,
        'Target_Met': 'Yes' if target_nonopt == actual_nonopt else f"No ({actual_nonopt}/{target_nonopt})"
    })

summary_df = pd.DataFrame(summary)
print(summary_df.to_string())

# Create a summary CSV
summary_csv = analysis_dir / "undersampling_summary.csv"
summary_df.to_csv(summary_csv, index=False)
print(f"\nSummary saved to: {summary_csv}")

# Check for datasets where target wasn't met
print("\n" + "="*60)
print("DATASETS WHERE TARGET NON-OPTIMAL SAMPLES WEREN'T REACHED:")
print("="*60)

problem_datasets = summary_df[
    (summary_df['Undersampled'] == 'Yes') & 
    (summary_df['Target_Non'].astype(int) > summary_df['Actual_Non'].astype(int))
]

if len(problem_datasets) > 0:
    for _, row in problem_datasets.iterrows():
        print(f"  {row['Dataset']}: target={row['Target_Non']}, actual={row['Actual_Non']}, optimal={row['Optimal']}")
    print(f"\nNote: This happens when clusters don't have enough samples to meet the target.")
    print(f"Check the analysis JSON files in {analysis_dir} for detailed cluster information.")
else:
    print("  All datasets reached their target non-optimal sample counts!")