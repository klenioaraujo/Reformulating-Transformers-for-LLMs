
import os
import subprocess
import re
import sys

# --- Funções Auxiliares ---

def run_command(command, description, capture=True):
    """Executa um comando do shell, imprime a descrição e lida com erros."""
    print(f"🚀 {description}...")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=capture,
            text=True,
            executable='/bin/bash'
        )
        print(f"✅ Sucesso: {description}")
        if capture and result.stdout:
            # Imprime apenas a última linha do output para ser mais conciso
            last_line = result.stdout.strip().split('\n')[-1]
            print(f"   Output: {last_line}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro: Falha em \'\'{\'description\'}\'\' com código de saída {e.returncode}.")
        print("   Sugestão: Verifique o comando e garanta que os caminhos e permissões estão corretos.")
        if capture and e.stderr:
            print(f"   Stderr: {e.stderr.strip()}")
        return False

def parse_requirements(input_file, output_file):
    """(Melhoria 1) Analisa robustamente um arquivo requirements.txt usando regex."""
    print(f"🔍 Analisando '{input_file}' para criar '{output_file}'...")
    try:
        with open(input_file, 'r') as f_in:
            lines = f_in.readlines()

        cleaned_packages = set()
        # Regex para capturar linhas de pacotes válidas, ignorando comentários e linhas vazias.
        # Lida com formatos como: package, package==version, package>=version, package[extra]
        package_regex = re.compile(r"^\s*([a-zA-Z0-9\-_]+(?:\\[a-zA-Z0-9\-_,]+\\])?(?:(?:==|>=|<=|~=)[a-zA-Z0-9\.\*]+)?)\s*(?:#.*)?$")

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or 'Makodev0' in line:
                continue

            # Lida com instalações editáveis ou de links git
            if line.startswith('-e') or line.startswith('git+'):
                 cleaned_packages.add(line)
                 continue

            match = package_regex.match(line)
            if match:
                # Remove especificadores de versão, como no script original
                package_name = re.split(r'[=><~]', match.group(1))[0].strip()
                if package_name:
                    cleaned_packages.add(package_name)

        # Aplica correção para o conhecido problema do python-dateutilpost0
        if "python-dateutilpost0" in cleaned_packages:
            print("🩹 Aplicando correção específica para 'python-dateutilpost0'...")
            cleaned_packages.remove("python-dateutilpost0")
            cleaned_packages.add("python-dateutil")
            print("✅ 'python-dateutilpost0' substituído por 'python-dateutil'.")

        with open(output_file, 'w') as f_out:
            f_out.write("\n".join(sorted(list(cleaned_packages))))

        print(f"✅ Análise concluída e '{output_file}' criado com sucesso.")
        return True
    except Exception as e:
        print(f"❌ Erro ao analisar o arquivo de dependências: {e}")
        return False

def check_and_install_missing(packages_to_install, requirements_file):
    """(Melhoria 3) Verifica pacotes instalados e instala apenas os que faltam."""
    print("📦 Verificando bibliotecas de ML ausentes...")
    try:
        installed_packages_raw = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']).decode()
        installed_packages = {p.split('==')[0].lower() for p in installed_packages_raw.splitlines()}

        with open(requirements_file, 'r') as f:
            required_packages = {line.strip().lower() for line in f if line.strip()}

        # Adiciona pacotes de ML à lista de requeridos se não estiverem lá
        for p in packages_to_install:
            required_packages.add(p.lower())

        missing_packages = [p for p in required_packages if p not in installed_packages]

        if not missing_packages:
            print("✅ Todas as bibliotecas necessárias já estão instaladas.")
            return True

        print(f"   Bibliotecas ausentes ou a serem atualizadas: {len(missing_packages)}. Instalando agora...")
        # Instala a partir do arquivo de requerimentos para garantir as versões corretas
        return run_command(f"{sys.executable} -m pip install -r {requirements_file}", "Instalando dependências de requirements_clean.txt")

    except Exception as e:
        print(f"❌ Erro ao verificar pacotes ausentes: {e}")
        return False


# --- Execução Principal ---

def main():
    """Função principal que orquestra a configuração do ambiente."""
    try:
        # O script agora é executado de dentro do repositório.
        # A clonagem e a entrada no diretório são de responsabilidade do usuário.

        # 1. Mudar para a branch correta
        if not run_command(
            "git checkout pure_physics_PsiQRH",
            "Mudando para a branch 'pure_physics_PsiQRH'"
        ):
            return

        # 3. (Melhoria 2) Verificar se os arquivos críticos existem
        print("🔎 Verificando a existência de arquivos críticos...")
        critical_files = ["benchmark_psiqrh.py", "psiqrh_pipeline.py"]
        if not all(os.path.exists(f) for f in critical_files):
            print(f"❌ Arquivo crítico não encontrado. Verifique se o repositório e a branch estão corretos.")
            return
        print("✅ Todos os arquivos críticos foram encontrados.")

        # 4. (Melhorias 1, 3, 4) Limpar e instalar dependências
        if not parse_requirements("requirements.txt", "requirements_clean.txt"):
            return

        ml_libs = ['datasets', 'evaluate', 'transformers', 'torch']
        if not check_and_install_missing(ml_libs, "requirements_clean.txt"):
             print("   Aviso: A instalação de dependências falhou. O script continuará, mas pode haver erros.")


        # 5. (Melhoria 6) Verificação de Status do Sistema
        print("🔎 Realizando verificação de status do sistema...")
        try:
            import torch
            import transformers
            print(f"   ✅ PyTorch versão: {torch.__version__}")
            print(f"   ✅ Transformers versão: {transformers.__version__}")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"   ✅ PyTorch está usando o dispositivo: {device}")
        except ImportError as e:
            print(f"❌ Falha na Verificação de Status: Não foi possível importar uma biblioteca crítica. Erro: {e}")
            return

        # 6. Executar o benchmark
        if not run_command(
            "python benchmark_psiqrh.py --benchmark glue --glue_task sst2",
            "Executando o benchmark GLUE sst2",
            capture=False # Mostrar output em tempo real
        ):
            return

        # 7. Testar importações finais
        if not run_command(
            "python -c \"try: from psiqrh_llm import PsiQRHConfig, PsiQRHForCausalLM; print('✅ Módulos ΨQRH OK') except Exception as e: print(f'❌ Erro: {e}')\"",
            "Testando importações do módulo final ΨQRH"
        ):
            return

        print("\n🎉🎉🎉 Todos os passos foram concluídos com sucesso! 🎉🎉🎉")

    except Exception as e:
        print(f"\n🚨 Um erro inesperado ocorreu durante a execução: {e}")
        print("   Por favor, revise os logs acima para diagnosticar o problema.")

# --- Executar a função principal ---
if __name__ == "__main__":
    main()
