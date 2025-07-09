import os
from huggingface_hub import snapshot_download
from rich.console import Console
from rich.panel import Panel
from transformers import AutoTokenizer, AutoModel

console = Console()

def download_qwen3_embedding_model_gui():
    model_id = "BAAI/bge-large-en-v1.5"
    local_dir = os.path.join(os.getcwd(), "models/BAAI/bge-large-en-v1.5")

    console.print(Panel.fit(f"[bold cyan]{model_id} Model Downloader[/bold cyan]"))

    console.print("[yellow]Starting model download... please wait.[/yellow]")

    # This will show tqdm in terminal, not rich (but still clean)
    model_dir = snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True
    )

    console.print("\n✅ [bold green]Download completed successfully![/bold green]")
    console.print(f"[cyan]Model files saved at:[/cyan] [bold]{model_dir}[/bold]")

    try:
        console.print("\n🔁 [yellow]Verifying tokenizer and model loading...[/yellow]")
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
        console.print("✅ [green]Model and tokenizer loaded successfully.[/green]")
    except Exception as e:
        console.print(f"❌ [red]Error loading model/tokenizer:[/red] {e}")

    console.print(f"\n📍 [bold]To load locally later:[/bold] [blue]AutoModel.from_pretrained('{model_dir}')[/blue]")

if __name__ == "__main__":
    download_qwen3_embedding_model_gui()
