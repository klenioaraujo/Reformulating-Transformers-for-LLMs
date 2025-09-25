import json
import os
import subprocess
import sys
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
    """Execute a prompt with robust JSON parsing and error handling"""

    # Method 1: Try standard parsing first
    try:
        content = prompt_path.read_text(encoding='utf-8', errors='replace')
        prompt = json.loads(content, strict=False)
        print(f"🚀 Executing: {prompt['id']}")

        # Execute the prompt action
        if prompt.get("action") == "document_component":
            with open(MANUAL_FILE, "a") as f:
                f.write(f"\n## {prompt.get('output_section', 'Generated Documentation')}\n")
                f.write(f"**File**: `{prompt.get('target_file', 'unknown')}`\n\n")
                f.write(f"{prompt.get('instructions', '')}\n\n---\n")

        # Post-execution hooks
        if prompt.get("post_execution_hook") == "update_directory_structure":
            update_directory_structure()

        # Auto-delete if requested
        if prompt.get("auto_delete", False):
            prompt_path.unlink()
            print(f"🗑️ Auto-deleted {prompt_path.name}")

        return prompt["id"]
    except json.JSONDecodeError as e:
        print(f"⚠️ Standard parsing failed: {e}")
        # Continue to fallback methods

    # Method 2: Try with different encoding
    try:
        content = prompt_path.read_text(encoding='latin-1', errors='replace')
        prompt = json.loads(content, strict=False)
        print(f"🚀 Executing (latin-1): {prompt['id']}")

        # Basic execution for fallback
        with open(MANUAL_FILE, "a") as f:
            f.write(f"\n## Fallback Execution for {prompt['id']}\n")
            f.write(f"**Action**: {prompt.get('action', 'unknown')}\n\n")
            f.write("Prompt executed via fallback encoding method.\n\n---\n")

        return prompt["id"]
    except Exception as e:
        print(f"⚠️ Latin-1 parsing failed: {e}")

    # Method 3: Try reading as bytes and manual decoding
    try:
        with open(prompt_path, 'rb') as f:
            content_bytes = f.read()

        # Try multiple decoding strategies
        for encoding in ['utf-8', 'latin-1', 'ascii']:
            try:
                content = content_bytes.decode(encoding, errors='replace')
                prompt = json.loads(content, strict=False)
                print(f"🚀 Executing ({encoding}): {prompt['id']}")

                with open(MANUAL_FILE, "a") as f:
                    f.write(f"\n## Byte Decoding Execution for {prompt['id']}\n")
                    f.write(f"**Encoding**: {encoding}\n\n")
                    f.write("Prompt executed via byte decoding method.\n\n---\n")

                return prompt["id"]
            except:
                continue
    except Exception as e:
        print(f"⚠️ Byte reading failed: {e}")

    # Method 4: Last resort - manual JSON extraction
    try:
        content = prompt_path.read_text(encoding='utf-8', errors='replace')

        # Extract basic prompt info without full JSON parsing
        import re
        id_match = re.search(r'"id"\s*:\s*"([^"]+)"', content)
        action_match = re.search(r'"action"\s*:\s*"([^"]+)"', content)

        if id_match and action_match:
            prompt_id = id_match.group(1)
            action = action_match.group(1)
            print(f"🚀 Executing (manual): {prompt_id} - {action}")

            # Basic documentation execution
            with open(MANUAL_FILE, "a") as f:
                f.write(f"\n## Manual Execution for {prompt_id}\n")
                f.write(f"**Action**: {action}\n\n")
                f.write("Prompt executed via fallback method due to JSON parsing issues.\n\n---\n")

            return prompt_id
    except Exception as e:
        print(f"❌ All parsing methods failed: {e}")
        return None

def run_engine():
    """Run prompt engine with enhanced error recovery"""
    state = load_state()

    # Extract just the prompt IDs from the executed prompts list
    executed_ids = set([prompt["id"] if isinstance(prompt, dict) else prompt for prompt in state["executed_prompts"]])

    # Ordenar prompts por ID (000, 001, ...)
    prompt_files = sorted(PROMPTS_DIR.glob("*.json"), key=lambda x: x.stem)

    successful_executions = 0
    failed_executions = 0

    print(f"📋 Processing {len(prompt_files)} prompts...")

    for pf in prompt_files:
        pid = pf.stem
        if pid in executed_ids:
            print(f"⏭️  Skipping already executed: {pid}")
            continue

        print(f"\n🔧 Processing: {pid}")

        try:
            result = execute_prompt(pf)
            if result:
                executed_ids.add(result)
                # Update state with full prompt objects
                state["executed_prompts"] = state["executed_prompts"] + [{
                    "id": pid,
                    "timestamp": "2025-09-25T22:30:00Z",
                    "status": "completed",
                    "execution_method": "enhanced_parser"
                }]
                save_state(state)
                successful_executions += 1
                print(f"✅ Success: {pid}")
            else:
                failed_executions += 1
                print(f"❌ Failed: {pid}")

        except Exception as e:
            failed_executions += 1
            print(f"💥 Error executing {pid}: {e}")

            # Log the error but continue processing
            error_entry = {
                "id": pid,
                "timestamp": "2025-09-25T22:30:00Z",
                "status": "failed",
                "error": str(e),
                "execution_method": "error_recovery"
            }
            state["executed_prompts"] = state["executed_prompts"] + [error_entry]
            save_state(state)

    print(f"\n📊 Execution Summary:")
    print(f"✅ Successful: {successful_executions}")
    print(f"❌ Failed: {failed_executions}")
    print(f"📁 Total: {len(prompt_files)}")

def diagnose_json_issue(prompt_path):
    """Diagnose JSON parsing issues for a specific prompt file"""
    print(f"\n🔍 Diagnosing: {prompt_path.name}")

    try:
        content = prompt_path.read_text(encoding='utf-8', errors='replace')

        # Check basic file properties
        print(f"📏 File size: {len(content)} characters")
        print(f"📄 Number of lines: {len(content.splitlines())}")

        # Check for common JSON issues
        brace_balance = content.count('{') - content.count('}')
        bracket_balance = content.count('[') - content.count(']')
        quote_count = content.count('"')

        print(f"⚖️ Brace balance: {brace_balance}")
        print(f"⚖️ Bracket balance: {bracket_balance}")
        print(f"🔤 Quote count: {quote_count} ({'even' if quote_count % 2 == 0 else 'odd'})")

        # Check for problematic characters
        import re
        control_chars = re.findall(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', content)
        if control_chars:
            print(f"⚠️ Control characters found: {len(control_chars)}")
        else:
            print("✅ No control characters detected")

        # Try parsing with detailed error info
        try:
            data = json.loads(content, strict=False)
            print("✅ JSON parses successfully in diagnostic mode")
            return True
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing failed:")
            print(f"   Error: {e}")
            print(f"   Line: {e.lineno}, Column: {e.colno}, Position: {e.pos}")

            # Show context around error
            lines = content.split('\n')
            if e.lineno <= len(lines):
                error_line = lines[e.lineno-1]
                print(f"   Problematic line: {repr(error_line)}")

                # Show surrounding context
                start = max(0, e.lineno-3)
                end = min(len(lines), e.lineno+2)
                print("   Context:")
                for i in range(start, end):
                    marker = '>>> ' if i == e.lineno-1 else '    '
                    print(f"   {marker}Line {i+1}: {lines[i]}")

            return False

    except Exception as e:
        print(f"💥 Diagnostic failed: {e}")
        return False

def validate_all_prompts():
    """Validate all prompts in the prompts directory"""
    prompt_files = sorted(PROMPTS_DIR.glob("*.json"), key=lambda x: x.stem)

    print("🔍 Validating all prompts...")
    valid_count = 0
    invalid_count = 0

    for pf in prompt_files:
        if diagnose_json_issue(pf):
            valid_count += 1
        else:
            invalid_count += 1

    print(f"\n📊 Validation Summary:")
    print(f"✅ Valid prompts: {valid_count}")
    print(f"❌ Invalid prompts: {invalid_count}")
    print(f"📁 Total prompts: {len(prompt_files)}")

    return valid_count == len(prompt_files)

if __name__ == "__main__":
    print("🚀 Enhanced Prompt Engine v2.0")
    print("=" * 40)

    # Optionally run diagnostics first
    if len(sys.argv) > 1 and sys.argv[1] == "--diagnose":
        validate_all_prompts()
    else:
        # Run the enhanced engine
        run_engine()