import json
import subprocess
import sys

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(f"{title:^80}")
    print("="*80 + "\n")

def run_script(script_name, description):
    """Run a script and print its output"""
    print(f"Running: {description}...")
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        print(e.stderr)

def main():
    print_section("THESIS RESULTS - COMPLETE SUMMARY")
    
    # 1. Baseline Self-Preference
    print_section("1. BASELINE SELF-PREFERENCE BIAS")
    run_script("print_baseline_bias.py", "Baseline bias metrics")
    
    # 2. Intervention Results
    print_section("2. INTERVENTION COMPARISON")
    run_script("print_intervention_comparison.py", "All intervention results")
    
    # 3. Classifier Accuracy
    print_section("3. CLASSIFIER ACCURACY")
    run_script("print_classifier_accuracy.py", "Classifier performance")
    
    # 4. Semantic Fidelity
    print_section("4. SEMANTIC FIDELITY (SBERT)")
    run_script("print_semantic_fidelity.py", "SBERT scores")
    
    # 5. Summary Table
    print_section("5. FINAL SUMMARY TABLE")
    run_script("print_summary_table.py", "Complete summary")
    
    print_section("ALL RESULTS PRINTED")
    print("Results saved to individual files in results/ directory")

if __name__ == "__main__":
    main()
