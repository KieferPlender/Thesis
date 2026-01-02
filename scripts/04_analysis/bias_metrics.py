import pandas as pd
import json
from typing import Tuple, Dict, List


def parse_winner(judge_response: str) -> str:
    """
    Parse the judge's response to extract the winner.
    
    Uses the LAST occurrence of [[A]]/[[B]]/[[C]] since judges often mention
    markers during analysis before giving their final verdict.
    
    Args:
        judge_response: The raw judge response text
        
    Returns:
        'model_a', 'model_b', 'tie', or 'error'
    """
    last_pos = -1
    winner = 'error'
    for marker, result in [('[[A]]', 'model_a'), ('[[B]]', 'model_b'), ('[[C]]', 'tie')]:
        pos = judge_response.rfind(marker)  # rfind = last occurrence
        if pos > last_pos:
            last_pos = pos
            winner = result
    return winner


def load_judge_results(results_file: str, metadata_file: str) -> pd.DataFrame:
    """
    Load judge results and merge with metadata.
    
    Args:
        results_file: Path to judge results JSONL (e.g., 'judge_results.jsonl')
        metadata_file: Path to metadata JSONL (e.g., 'judge_samples.jsonl')
        
    Returns:
        DataFrame with columns: conversation_id, judge_model, winner, 
                                model_a_name, model_b_name, human_winner
    """
    # Load results
    results = []
    with open(results_file, 'r') as f:
        for line in f:
            try:
                row = json.loads(line)
                row['winner'] = parse_winner(row.get('judge_response', ''))
                results.append(row)
            except:
                pass
    results_df = pd.DataFrame(results)
    
    # Load metadata
    metadata = []
    with open(metadata_file, 'r') as f:
        for line in f:
            try:
                metadata.append(json.loads(line))
            except:
                pass
    meta_df = pd.DataFrame(metadata)[['conversation_id', 'model_a_name', 'model_b_name', 'human_winner']]
    
    # Merge
    return results_df.merge(meta_df, on='conversation_id', how='left')


def calculate_biased_self_preference(df: pd.DataFrame, judge_name: str) -> Dict[str, float]:
    """
    Calculate biased self-preference metrics for a specific judge.
    
    Biased self-preference = How often the judge picks itself when humans picked the opponent.
    This is the key bias metric.
    
    Args:
        df: DataFrame with judge results (from load_judge_results)
        judge_name: Name of the judge model (e.g., 'deepseek-r1-0528')
        
    Returns:
        Dictionary with:
            - bias_rate: Percentage (0-100) of biased picks
            - biased_picks: Number of times judge picked itself when human picked opponent
            - opportunities: Number of times human picked opponent (denominator)
            - total_self_judgments: Total battles where judge evaluated itself
    """
    # Filter: Judge is the judge AND judge is a contestant
    df_judge = df[
        (df['judge_model'] == judge_name) &
        ((df['model_a_name'] == judge_name) | (df['model_b_name'] == judge_name))
    ]
    
    total_self_judgments = len(df_judge)
    
    if total_self_judgments == 0:
        return {
            'bias_rate': 0.0,
            'biased_picks': 0,
            'opportunities': 0,
            'total_self_judgments': 0
        }
    
    # Count biased picks
    biased_picks = 0
    opportunities = 0
    
    for _, row in df_judge.iterrows():
        judge_is_a = (row['model_a_name'] == judge_name)
        
        # Did human pick opponent?
        human_picked_opponent = (
            (judge_is_a and row['human_winner'] == 'model_b') or
            (not judge_is_a and row['human_winner'] == 'model_a')
        )
        
        if human_picked_opponent:
            opportunities += 1
            
            # Did judge override and pick itself?
            judge_picked_self = (
                (judge_is_a and row['winner'] == 'model_a') or
                (not judge_is_a and row['winner'] == 'model_b')
            )
            
            if judge_picked_self:
                biased_picks += 1
    
    bias_rate = (biased_picks / opportunities * 100) if opportunities > 0 else 0.0
    
    return {
        'bias_rate': bias_rate,
        'biased_picks': biased_picks,
        'opportunities': opportunities,
        'total_self_judgments': total_self_judgments
    }


def calculate_raw_self_preference(df: pd.DataFrame, judge_name: str) -> Dict[str, float]:
    """
    Calculate raw self-preference (how often judge picks itself, regardless of human opinion).
    
    This is a simpler metric but doesn't account for whether the judge was correct.
    
    Args:
        df: DataFrame with judge results
        judge_name: Name of the judge model
        
    Returns:
        Dictionary with:
            - self_preference_rate: Percentage of times judge picked itself
            - self_picks: Number of times judge picked itself
            - total_self_judgments: Total battles where judge evaluated itself
    """
    df_judge = df[
        (df['judge_model'] == judge_name) &
        ((df['model_a_name'] == judge_name) | (df['model_b_name'] == judge_name))
    ]
    
    total = len(df_judge)
    if total == 0:
        return {'self_preference_rate': 0.0, 'self_picks': 0, 'total_self_judgments': 0}
    
    self_picks = 0
    for _, row in df_judge.iterrows():
        judge_is_a = (row['model_a_name'] == judge_name)
        judge_picked_self = (
            (judge_is_a and row['winner'] == 'model_a') or
            (not judge_is_a and row['winner'] == 'model_b')
        )
        if judge_picked_self:
            self_picks += 1
    
    return {
        'self_preference_rate': (self_picks / total * 100),
        'self_picks': self_picks,
        'total_self_judgments': total
    }


def compare_interventions(baseline_df: pd.DataFrame, 
                         intervention_df: pd.DataFrame, 
                         judge_name: str) -> Dict[str, any]:
    """
    Compare bias metrics between baseline and intervention for a specific judge.
    
    Args:
        baseline_df: DataFrame with baseline results
        intervention_df: DataFrame with intervention results
        judge_name: Name of the judge model
        
    Returns:
        Dictionary with baseline metrics, intervention metrics, and changes
    """
    baseline_metrics = calculate_biased_self_preference(baseline_df, judge_name)
    intervention_metrics = calculate_biased_self_preference(intervention_df, judge_name)
    
    # Calculate changes
    bias_reduction = baseline_metrics['bias_rate'] - intervention_metrics['bias_rate']
    relative_reduction = (bias_reduction / baseline_metrics['bias_rate'] * 100) if baseline_metrics['bias_rate'] > 0 else 0
    
    return {
        'judge': judge_name,
        'baseline': baseline_metrics,
        'intervention': intervention_metrics,
        'bias_reduction_pp': bias_reduction,  # percentage points
        'bias_reduction_relative': relative_reduction  # relative %
    }


def get_verdict_changes(baseline_df: pd.DataFrame, 
                       intervention_df: pd.DataFrame, 
                       judge_name: str) -> Dict[str, int]:
    """
    Count how many verdicts changed between baseline and intervention.
    
    Args:
        baseline_df: DataFrame with baseline results
        intervention_df: DataFrame with intervention results
        judge_name: Name of the judge model
        
    Returns:
        Dictionary with counts of different types of changes
    """
    # Get overlapping battles
    base_judge = baseline_df[
        (baseline_df['judge_model'] == judge_name) &
        ((baseline_df['model_a_name'] == judge_name) | (baseline_df['model_b_name'] == judge_name))
    ]
    
    inter_judge = intervention_df[
        (intervention_df['judge_model'] == judge_name) &
        ((intervention_df['model_a_name'] == judge_name) | (intervention_df['model_b_name'] == judge_name))
    ]
    
    base_ids = set(base_judge['conversation_id'])
    inter_ids = set(inter_judge['conversation_id'])
    overlap = base_ids & inter_ids
    
    # Count changes
    total_changed = 0
    bias_reduced = 0
    bias_increased = 0
    
    for conv_id in overlap:
        base_row = base_judge[base_judge['conversation_id'] == conv_id].iloc[0]
        inter_row = inter_judge[inter_judge['conversation_id'] == conv_id].iloc[0]
        
        if base_row['winner'] != inter_row['winner']:
            total_changed += 1
            
            # Check if this affected bias
            judge_is_a = (base_row['model_a_name'] == judge_name)
            human_picked_opponent = (
                (judge_is_a and base_row['human_winner'] == 'model_b') or
                (not judge_is_a and base_row['human_winner'] == 'model_a')
            )
            
            if human_picked_opponent:
                base_picked_self = (
                    (judge_is_a and base_row['winner'] == 'model_a') or
                    (not judge_is_a and base_row['winner'] == 'model_b')
                )
                inter_picked_self = (
                    (judge_is_a and inter_row['winner'] == 'model_a') or
                    (not judge_is_a and inter_row['winner'] == 'model_b')
                )
                
                if base_picked_self and not inter_picked_self:
                    bias_reduced += 1
                elif not base_picked_self and inter_picked_self:
                    bias_increased += 1
    
    return {
        'total_battles': len(overlap),
        'verdicts_changed': total_changed,
        'bias_reduced_count': bias_reduced,
        'bias_increased_count': bias_increased,
        'net_improvement': bias_reduced - bias_increased
    }
