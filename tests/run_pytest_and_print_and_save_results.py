from datetime import datetime
from tabulate import tabulate
import re
from typing import List, Dict
import subprocess
import os


def parse_pytest_output(output: str) -> List[Dict[str, any]]:
    # Match lines like:
    # all_llm_provider/test_all_llm_provider.py .......... [55%]
    # llamaindex_examples/legal_research_rag/test_legal_rag.py F [94%]
    test_result_pattern = re.compile(r"^(.*\.py)\s+([.EF]+)")
    results = []
    for line in output.splitlines():
        match = test_result_pattern.match(line.strip())
        if match:
            module = match.group(1)
            result_str = match.group(2)
            passed = result_str.count(".")
            failed = result_str.count("F")
            errors = result_str.count("E")
            total = len(result_str)
            results.append({
                "module": module,
                "count": total,
                "passed": passed,
                "failed": failed,
                "errors": errors
            })
    return results


def generate_test_report(test_results, duration):
    total_tests = sum(item["count"] for item in test_results)
    total_passed = sum(item["passed"] for item in test_results)
    total_failed = sum(item["failed"] for item in test_results)
    total_errors = sum(item["errors"] for item in test_results)
    summary = f"""
TEST EXECUTION REPORT
=====================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Summary:
- Duration: {duration}
- Total Tests: {total_tests}
- Passed: {total_passed} ({total_passed / total_tests * 100:.1f}%)
- Failed: {total_failed} ({total_failed / total_tests * 100:.1f}%)
- Errors: {total_errors} ({total_errors / total_tests * 100:.1f}%)
"""
    # Create rows for tabulate
    table_data = []
    for result in test_results:
        if result["errors"] > 0:
            status = "💥"  # Error symbol
        elif result["failed"] > 0:
            status = "❌"  # Failed symbol
        else:
            status = "✅"  # Passed symbol
            
        table_data.append([
            result["module"],
            result["count"],
            result["passed"],
            result["failed"],
            result["errors"],
            status
        ])
    headers = ["Test Module", "Tests", "Passed", "Failed", "Errors", "Status"]
    table = tabulate(table_data, headers=headers, tablefmt="fancy_grid", 
                    colalign=("left", "right", "right", "right", "right", "center"))
    report = summary + "\nDetailed Test Results:\n" + table
    
    if total_failed > 0 or total_errors > 0:
        problematic_tests = [r for r in test_results if r["failed"] > 0 or r["errors"] > 0]
        report += "\n\nProblematic Tests:\n"
        for test in problematic_tests:
            issues = []
            if test["failed"] > 0:
                issues.append(f"{test['failed']} failed")
            if test["errors"] > 0:
                issues.append(f"{test['errors']} errors")
            report += f"- {test['module']}: {', '.join(issues)}\n"
        report += f"{'-'*50}\n"
        report += "  (Investigation needed - check test logs for specific issues)\n"
        report += f"{'-'*50}"
    return report


def save_report(report, filename=None):
    """Save the report to a file."""
    if filename is None:
        filename = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(filename, 'w') as file:
        file.write(report)
    print(f"Report saved to {os.path.abspath(filename)}")


def run_pytest_and_generate_report():
    start_time = datetime.now()

    # Run pytest
    output = subprocess.run(
        "python -m pytest",
        shell=True,
        capture_output=True,
        text=True
    ).stdout

    # duration
    end_time = datetime.now()
    duration = f"{(end_time - start_time).total_seconds() / 60:.2f} minutes"


    # Parse test results from output
    test_results = parse_pytest_output(output)
    # Generate report
    report = generate_test_report(test_results, duration)
    # Print and save
    print(report)
    save_report(report)


if __name__ == "__main__":
    run_pytest_and_generate_report()
  