# model_test_loading.py
import time
from rich.console import Console
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc
import torch
import torch.version
from accelerate import infer_auto_device_map

cuda_enabled = torch.cuda.is_available()
cuda_device = torch.cuda.current_device() if cuda_enabled else None
cuda_version = torch.version.cuda if cuda_enabled else None
console = Console()

def load_model():
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    model_dir = "C:\\Users\\MayanksPotato\\Desktop\\DMRC_Chatbot\\chatbot\\models\\meta-llama\\Llama-3.2-3B-Instruct"
    try:
        console.print("\n🔁 [yellow]Verifying tokenizer and model loading...[/yellow]")
        tokenizer = AutoTokenizer.from_pretrained(model_dir , trust_remote_code=True)
        console.print("✅ [green]Tokenizer loaded successfully.[/green]")
        console.print("EOS token:", tokenizer.eos_token)
        console.print("BOS token:", tokenizer.bos_token)
        console.print("UNK token:", tokenizer.unk_token)
        console.print("PAD token:", tokenizer.pad_token)
        console.print("Model ID:", model_id)
        max_memory = {
            0: "6GiB",  # ✅ Key is int (GPU ID), value is str (allowed)
            "cpu": "6GiB"  # ✅ Key is str, value is str — both acceptable
        }

        # First load the model without device mapping
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=None  # Don't map devices yet
        )
        
        # Then infer device map
        device_map = infer_auto_device_map(
            model,
            max_memory=max_memory, # type: ignore
            no_split_module_classes=["LlamaDecoderLayer"]
        )
        
        # Apply the device map using accelerate's dispatch_model
        from accelerate import dispatch_model
        model = dispatch_model(model, device_map=device_map)

        console.print("✅ [green]Model and tokenizer loaded successfully.[/green]")
        console.print("[yellow]Waiting for 30 seconds...[/yellow]")
        time.sleep(30)
        console.print("[green]30 seconds have passed.[/green]")
        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        console.print("🧹 [blue]Model and tokenizer unloaded from RAM.[/blue]")
    except Exception as e:
        console.print(f"❌ [red]Error loading model/tokenizer:[/red] {e}")

if __name__ == "__main__":
    console.print(f"[bold cyan]CUDA Enabled:[/bold cyan] {cuda_enabled}")
    if cuda_enabled:
        console.print(f"[bold cyan]Current CUDA Device:[/bold cyan] {cuda_device}")
        console.print(f"[bold cyan]CUDA Version:[/bold cyan] {cuda_version}")
    else:
        console.print("[bold red]CUDA is not available.[/bold red]")
    
    load_model()