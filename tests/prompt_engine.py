import json
import os
import subprocess
from pathlib import Path

MANUAL_DIR = Path("construction_technical_manual")
PROMPTS_DIR = MANUAL_DIR / "prompts"
STATE_FILE = MANUAL_DIR / "state.json"
MANUAL_FILE = MANUAL_DIR / "manual.md"
STRUCTURE_FILE = Path("estrutura_diretorios.txt")

def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"executed_prompts": [], "manual_version": "0.0.0"}

def save_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))

def update_directory_structure():
    # Gera estrutura atual do repo
    result = subprocess.run(["tree", "-d", "-L", "3"], capture_output=True, text=True)
    STRUCTURE_FILE.write_text(result.stdout)
    print(f"✅ Updated {STRUCTURE_FILE}")

def execute_prompt(prompt_path):
    prompt = json.loads(prompt_path.read_text())
    print(f"🚀 Executing: {prompt['id']}")

    # Aqui você pode:
    # - Chamar uma LLM local (ex: Ollama, vLLM) com o prompt['instructions']
    # - Ou executar um script Python personalizado
    # - Ou simplesmente anexar ao manual (para prompts documentais)

    if prompt["action"] == "document_component":
        # Simples exemplo: anexar instruções ao manual
        with open(MANUAL_FILE, "a") as f:
            f.write(f"\n## {prompt['output_section']}\n")
            f.write(f"**File**: `{prompt['target_file']}`\n\n")
            f.write(f"{prompt['instructions']}\n\n---\n")

    # Hooks pós-execução
    if prompt.get("post_execution_hook") == "update_directory_structure":
        update_directory_structure()

    # Auto-deletar?
    if prompt.get("auto_delete", False):
        prompt_path.unlink()
        print(f"🗑️ Auto-deleted {prompt_path.name}")

    return prompt["id"]

def run_engine():
    state = load_state()
    executed = set(state["executed_prompts"])

    # Ordenar prompts por ID (000, 001, ...)
    prompt_files = sorted(PROMPTS_DIR.glob("*.json"), key=lambda x: x.stem)

    for pf in prompt_files:
        pid = pf.stem
        if pid in executed:
            continue
        executed.add(execute_prompt(pf))
        state["executed_prompts"] = sorted(executed)
        save_state(state)

if __name__ == "__main__":
    run_engine()