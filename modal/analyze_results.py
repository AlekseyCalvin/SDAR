"""
Download and analyze GSM8K benchmark results from Modal volume.

Usage:
    modal run analyze_results.py --results-file gsm8k_JetLM_SDAR-4B-Chat_20251117_151516.json
"""
import modal
import json
import re

app = modal.App("analyze-gsm8k-results")

# Results volume
results_volume = modal.Volume.from_name("sdar-results", create_if_missing=False)
RESULTS_DIR = "/results"


@app.function(
    volumes={RESULTS_DIR: results_volume},
)
def analyze_results(model_folder: str, results_filename: str):
    """
    Analyze wrong answers from GSM8K benchmark results.

    Args:
        model_folder: Model folder name (e.g., "JetLM_SDAR-4B-Chat")
        results_filename: Results filename (e.g., "gsm8k_JetLM_SDAR-4B-Chat_20251117_151516.json")
    """
    import os

    results_path = f"{RESULTS_DIR}/{model_folder}/{results_filename}"

    if not os.path.exists(results_path):
        print(f"Error: File not found: {results_path}")
        print(f"Available files in {RESULTS_DIR}/{model_folder}:")
        if os.path.exists(f"{RESULTS_DIR}/{model_folder}"):
            for f in os.listdir(f"{RESULTS_DIR}/{model_folder}"):
                print(f"  - {f}")
        return None

    # Read results
    with open(results_path, 'r') as f:
        data = json.load(f)

    # Print config
    print("=" * 80)
    print("BENCHMARK CONFIGURATION")
    print("=" * 80)
    if 'config' in data:
        for key, val in data['config'].items():
            print(f"{key}: {val}")
    print()

    # Print metrics
    print("=" * 80)
    print("OVERALL METRICS")
    print("=" * 80)
    if 'metrics' in data:
        for key, val in data['metrics'].items():
            print(f"{key}: {val}")
    print()

    results = data.get('results', [])
    wrong_results = [r for r in results if not r.get('correct', False)]

    print(f"Total results: {len(results)}")
    print(f"Wrong answers: {len(wrong_results)}")
    print(f"Correct answers: {len(results) - len(wrong_results)}")
    print()

    # Categorize extraction issues
    issues = {
        'null_or_empty': [],
        'non_numeric': [],
        'partial_latex': [],
        'has_backslash': [],
        'has_dollar': [],
        'clean_number': []
    }

    def is_clean_number(s):
        if s is None or s == '':
            return False
        # Remove common number formatting
        cleaned = str(s).replace(',', '').replace('.', '').replace('-', '').replace(' ', '')
        return cleaned.isdigit() or (cleaned.replace('.', '', 1).isdigit())

    for r in wrong_results:
        pred = r.get('predicted_answer')

        if pred is None or pred == '':
            issues['null_or_empty'].append(r)
        elif '\\' in str(pred):
            issues['has_backslash'].append(r)
            if '\\frac' in str(pred) or '\\boxed' in str(pred):
                issues['partial_latex'].append(r)
        elif '$' in str(pred):
            issues['has_dollar'].append(r)
        elif not is_clean_number(pred):
            issues['non_numeric'].append(r)
        else:
            issues['clean_number'].append(r)

    print("=" * 80)
    print("EXTRACTION ISSUE CATEGORIES")
    print("=" * 80)
    print(f"Null/Empty: {len(issues['null_or_empty'])}")
    print(f"Has backslash (LaTeX escape): {len(issues['has_backslash'])}")
    print(f"Partial LaTeX commands: {len(issues['partial_latex'])}")
    print(f"Has dollar sign: {len(issues['has_dollar'])}")
    print(f"Non-numeric text: {len(issues['non_numeric'])}")
    print(f"Clean number (model error, not extraction): {len(issues['clean_number'])}")
    print()

    # Calculate actual accuracy if we fix extraction issues
    extraction_errors = (len(issues['null_or_empty']) +
                        len(issues['partial_latex']) +
                        len(issues['has_backslash']) +
                        len(issues['non_numeric']))

    model_errors = len(issues['clean_number'])

    print(f"Extraction errors: {extraction_errors}")
    print(f"Model errors (wrong answer with clean extraction): {model_errors}")
    print(f"Extraction error rate: {100*extraction_errors/len(wrong_results):.1f}% of wrong answers")
    print()

    # Show examples
    print("=" * 80)
    print("EXAMPLES OF EXTRACTION ISSUES")
    print("=" * 80)

    categories_to_show = ['null_or_empty', 'partial_latex', 'has_backslash', 'non_numeric', 'clean_number']
    for cat in categories_to_show:
        if issues[cat]:
            print(f"\n{cat.upper().replace('_', ' ')} - {len(issues[cat])} cases:")
            print("-" * 80)
            for i, example in enumerate(issues[cat][:5]):  # Show first 5
                idx = example.get('eval_number', example.get('idx', '?'))
                pred = example.get('predicted_answer', 'N/A')
                gt_answer = example.get('ground_truth_answer', 'N/A')

                # Get a snippet of generated text
                gen_text = example.get('generated_text_full', example.get('generated_text', ''))
                if gen_text:
                    # Show last 150 chars where the answer should be
                    snippet = gen_text[-150:] if len(gen_text) > 150 else gen_text
                else:
                    snippet = "N/A"

                print(f"\nExample {i+1} (eval #{idx}):")
                print(f"  Predicted: {repr(pred)}")
                print(f"  Ground truth: {repr(gt_answer)}")
                print(f"  Generated (end): ...{snippet}")

    print("\n" + "=" * 80)

    # Save analysis locally
    analysis_output = {
        'config': data.get('config', {}),
        'metrics': data.get('metrics', {}),
        'analysis': {
            'total_results': len(results),
            'correct': len(results) - len(wrong_results),
            'wrong': len(wrong_results),
            'extraction_issues': {
                'null_or_empty': len(issues['null_or_empty']),
                'has_backslash': len(issues['has_backslash']),
                'partial_latex': len(issues['partial_latex']),
                'has_dollar': len(issues['has_dollar']),
                'non_numeric': len(issues['non_numeric']),
                'clean_number': len(issues['clean_number']),
            },
            'extraction_error_rate': extraction_errors / len(wrong_results) if wrong_results else 0,
        },
        'wrong_examples': wrong_results[:20]  # First 20 wrong answers
    }

    return analysis_output


@app.local_entrypoint()
def main(
    model_folder: str = "JetLM_SDAR-4B-Chat",
    results_filename: str = "gsm8k_JetLM_SDAR-4B-Chat_20251117_151516.json",
    output_file: str = "results_analysis.json",
):
    """
    Download and analyze results from Modal volume.
    """
    print(f"Analyzing: {model_folder}/{results_filename}")
    print()

    analysis = analyze_results.remote(model_folder, results_filename)

    if analysis:
        # Save analysis locally
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\nAnalysis saved to {output_file}")
