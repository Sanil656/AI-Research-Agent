"""
Interactive CLI for the LangGraph Research AI Agent.
Runs deep research cycles and outputs a structured Markdown report.
"""

import sys
import os
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

# Ensure UTF-8 output encoding across Windows shells
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from agent import create_research_graph

console = Console()


def run_research(topic: str, max_iterations: int = 3, output_path: str = "research_report.md", provider: str = None, model: str = None):
    # Set provider/model overrides in environment if provided
    if provider:
        os.environ["DEFAULT_LLM_PROVIDER"] = provider
    if model:
        if provider == "openai":
            os.environ["OPENAI_MODEL"] = model
        elif provider == "gemini":
            os.environ["GEMINI_MODEL"] = model
        elif provider == "groq":
            os.environ["GROQ_MODEL"] = model

    console.print(Panel.fit(
        f"[bold cyan]LangGraph Autonomous Research AI Agent[/bold cyan]\n"
        f"[yellow]Topic:[/yellow] {topic}\n"
        f"[yellow]Max Research Depth Cycles:[/yellow] {max_iterations}",
        border_style="cyan"
    ))

    try:
        agent = create_research_graph()
    except Exception as e:
        console.print(f"[bold red]Initialization Error:[/bold red] {e}")
        return

    initial_state = {
        "topic": topic,
        "max_iterations": max_iterations,
        "iteration": 0,
        "plan": [],
        "queries": [],
        "findings": [],
        "reflection": "",
        "is_sufficient": False,
        "final_report": ""
    }

    final_report_content = ""

    console.print("[dim]Starting graph execution...[/dim]\n")

    try:
        # Stream updates from the graph step-by-step
        for step in agent.stream(initial_state, stream_mode="updates"):
            for node_name, output in step.items():
                if node_name == "plan":
                    plan_items = output.get("plan", [])
                    queries = output.get("queries", [])
                    console.print(Panel(
                        "\n".join([f"• {item}" for item in plan_items]) +
                        "\n\n[bold yellow]Initial Search Queries:[/bold yellow]\n" +
                        "\n".join([f"🔍 {q}" for q in queries]),
                        title="[bold green]🧠 1. Research Strategy & Deconstruction[/bold green]",
                        border_style="green"
                    ))

                elif node_name == "search":
                    findings = output.get("findings", [])
                    console.print(f"[bold blue]🔎 Search Completed:[/bold blue] Gathered {len(findings)} new evidence snippets.")

                elif node_name == "reflect":
                    critique = output.get("reflection", "")
                    is_sufficient = output.get("is_sufficient", False)
                    next_queries = output.get("queries", [])
                    iter_num = output.get("iteration", 1) - 1

                    status_tag = "[bold green]SUFFICIENT DEPTH ACHIEVED[/bold green]" if is_sufficient else f"[bold yellow]NEEDS DEEPER INVESTIGATION (Cycle {iter_num}/{max_iterations})[/bold yellow]"
                    
                    details = f"[bold]Status:[/bold] {status_tag}\n\n[bold]Critique & Gap Analysis:[/bold]\n{critique}"
                    if next_queries and not is_sufficient:
                        details += "\n\n[bold]Follow-up Inquiries:[/bold]\n" + "\n".join([f"➔ {q}" for q in next_queries])

                    console.print(Panel(
                        details,
                        title=f"[bold magenta]🪞 2. Deep Reflection & Gap Assessment (Cycle {iter_num})[/bold magenta]",
                        border_style="magenta"
                    ))

                elif node_name == "synthesize":
                    final_report_content = output.get("final_report", "")
                    console.print(Panel(
                        "[bold green]Synthesizing complete! Final report generated successfully.[/bold green]",
                        border_style="green"
                    ))

    except KeyboardInterrupt:
        console.print("\n[bold red]Research interrupted by user.[/bold red]")
        return
    except Exception as e:
        console.print(f"\n[bold red]Error during research execution:[/bold red] {e}")
        import traceback
        console.print(traceback.format_exc())
        return

    # Save final report to file
    if final_report_content:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_report_content)
        console.print(f"\n[bold cyan]Saved research report to:[/bold cyan] [underline]{os.path.abspath(output_path)}[/underline]\n")
        console.print(Panel(Markdown(final_report_content[:1000] + "\n\n*...[Content truncated for preview. Full report saved to file]...*"), title="Report Preview", border_style="cyan"))
    else:
        console.print("[yellow]No report was generated.[/yellow]")


def main():
    parser = argparse.ArgumentParser(description="LangGraph Research AI Agent - Deep Thinking without Constraints")
    parser.add_argument("--topic", "-t", type=str, help="Research topic or question")
    parser.add_argument("--max-iterations", "-i", type=int, default=3, help="Maximum thinking/research cycles (default: 3)")
    parser.add_argument("--output", "-o", type=str, default="research_report.md", help="Output file path for the report")
    parser.add_argument("--provider", "-p", type=str, choices=["gemini", "openai", "groq", "ollama"], help="LLM Provider")
    parser.add_argument("--model", "-m", type=str, help="Specific model name")

    args = parser.parse_args()

    topic = args.topic
    if not topic:
        console.print("[bold yellow]Enter your research topic or question:[/bold yellow] ", end="")
        try:
            topic = input().strip()
        except EOFError:
            topic = ""

    if not topic:
        console.print("[red]A research topic is required. Exiting.[/red]")
        sys.exit(1)

    run_research(
        topic=topic,
        max_iterations=args.max_iterations,
        output_path=args.output,
        provider=args.provider,
        model=args.model
    )


if __name__ == "__main__":
    main()
