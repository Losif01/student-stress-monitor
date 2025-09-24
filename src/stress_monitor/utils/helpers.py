from rich.console import Console
from rich.table import Table
from rich import print as rprint
from typing import Any, Dict

console = Console()

def print_results(results: Dict[str, Any]) -> None:
    """Print results using rich formatting"""
    table = Table(title="Model Results")
    table.add_column("Model", style="cyan")
    table.add_column("Accuracy", style="green")
    table.add_column("Parameters", style="yellow")
    
    for model_name, result in results.items():
        accuracy = result.accuracy if hasattr(result, 'accuracy') else result
        params = result.parameters if hasattr(result, 'parameters') else {}
        param_str = ", ".join([f"{k}={v}" for k, v in list(params.items())[:2]])
        table.add_row(model_name, f"{accuracy:.4f}", param_str)
    
    console.print(table)

def format_percentage(value: float) -> str:
    """Format float as percentage"""
    return f"{value:.2%}"