# Manual Steps Required

Algumas etapas precisam ser executadas manualmente por você (requerem sudo ou credenciais):

## 🔧 1. Corrigir Permissões do Diretório models/

O diretório `models/` pertence ao root. Execute:

```bash
sudo ./scripts/fix_permissions.sh
```

**Ou manualmente:**
```bash
sudo chown -R $USER:$USER models/
sudo chmod -R u+rw models/
```

**Verificar:**
```bash
ls -la models/
# Deve mostrar seu usuário como owner
```

---

## 📦 2. Reinstalar Pacote (Após Corrigir Permissões)

```bash
source .venv/bin/activate
pip install -e . --force-reinstall --no-cache-dir
```

**Verificar instalação:**
```bash
pip show psiqrh
# Deve mostrar: Name: psiqrh, Version: 1.0.0
```

---

## 🚀 3. Testar Build PyPI

```bash
source .venv/bin/activate
python -m build
twine check dist/*
```

**Resultado esperado:**
```
Successfully built psiqrh-1.0.0.tar.gz and psiqrh-1.0.0-py3-none-any.whl
Checking dist/psiqrh-1.0.0-py3-none-any.whl: PASSED
Checking dist/psiqrh-1.0.0.tar.gz: PASSED
```

---

## 🧪 4. Publicar no TestPyPI (Opcional mas Recomendado)

### 4.1. Criar Conta TestPyPI

1. Acesse: https://test.pypi.org/account/register/
2. Confirme email
3. Ative 2FA (obrigatório)

### 4.2. Gerar Token

1. Acesse: https://test.pypi.org/manage/account/token/
2. Crie token com escopo "Entire account"
3. Copie o token (começa com `pypi-`)

### 4.3. Configurar Credenciais

**Opção 1: Arquivo ~/.pypirc**
```bash
cat > ~/.pypirc << 'EOF'
[testpypi]
  username = __token__
  password = pypi-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

[pypi]
  username = __token__
  password = pypi-YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY
EOF

chmod 600 ~/.pypirc
```

**Opção 2: Variável de Ambiente**
```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

### 4.4. Upload para TestPyPI

```bash
./scripts/publish_to_pypi.sh test
```

**Ou manualmente:**
```bash
twine upload --repository testpypi dist/*
```

### 4.5. Testar Instalação do TestPyPI

```bash
# Em outro ambiente
python -m venv test_env
source test_env/bin/activate
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ psiqrh

# Testar
python -c "import psiqrh; print(psiqrh.__version__)"
```

---

## 🎯 5. Publicar no PyPI Real (Quando Estiver Pronto)

### 5.1. Criar Conta PyPI

1. Acesse: https://pypi.org/account/register/
2. Confirme email
3. Ative 2FA (obrigatório)

### 5.2. Gerar Token PyPI

1. Acesse: https://pypi.org/manage/account/token/
2. Crie token com escopo "Entire account" (ou específico para psiqrh)
3. Adicione ao ~/.pypirc (seção [pypi])

### 5.3. Upload

```bash
./scripts/publish_to_pypi.sh
```

**Ou manualmente:**
```bash
twine upload dist/*
```

### 5.4. Verificar

Acesse: https://pypi.org/project/psiqrh/

Instale:
```bash
pip install psiqrh
```

---

## 📋 Checklist de Execução

Execute nesta ordem:

- [ ] 1. `sudo ./scripts/fix_permissions.sh`
- [ ] 2. Verificar: `ls -la models/` (deve ser seu usuário)
- [ ] 3. `source .venv/bin/activate`
- [ ] 4. `pip install -e . --force-reinstall`
- [ ] 5. Verificar: `pip show psiqrh`
- [ ] 6. `python -m build`
- [ ] 7. `twine check dist/*`
- [ ] 8. Criar conta TestPyPI
- [ ] 9. Gerar token TestPyPI
- [ ] 10. Configurar ~/.pypirc
- [ ] 11. `./scripts/publish_to_pypi.sh test`
- [ ] 12. Testar instalação do TestPyPI
- [ ] 13. Se OK, criar conta PyPI
- [ ] 14. Gerar token PyPI
- [ ] 15. `./scripts/publish_to_pypi.sh`
- [ ] 16. Verificar em https://pypi.org/project/psiqrh/

---

## 🆘 Troubleshooting

### Erro: "Permission denied" em models/

```bash
sudo ./scripts/fix_permissions.sh
```

### Erro: "Package already exists on PyPI"

Se já publicou, incremente a versão em `pyproject.toml`:
```toml
version = "1.0.1"  # Era 1.0.0
```

Rebuild:
```bash
rm -rf dist/ build/ *.egg-info
python -m build
```

### Erro: "Invalid token"

1. Regenere o token no site (TestPyPI ou PyPI)
2. Atualize ~/.pypirc
3. Ou use variável de ambiente:
   ```bash
   export TWINE_PASSWORD=pypi-NOVO-TOKEN
   ```

### Erro: "403 Forbidden"

- Verifique se 2FA está ativado na conta
- Use token (não senha)
- Token deve ter permissões corretas

### Build demora muito / falha

```bash
# Limpar cache
rm -rf ~/.cache/pip
pip cache purge

# Rebuild
python -m build --no-isolation
```

---

## ✅ Quando Tudo Estiver Publicado

1. **Adicione badge ao README:**
   ```markdown
   [![PyPI](https://img.shields.io/pypi/v/psiqrh)](https://pypi.org/project/psiqrh/)
   ```

2. **Crie release no GitHub:**
   ```bash
   git tag -a v1.0.0 -m "FAIR-compliant release 1.0.0"
   git push origin v1.0.0
   ```

3. **Atualize Zenodo:**
   - Acesse: https://zenodo.org/records/17171112
   - Adicione link para PyPI
   - Atualize metadados

4. **Anuncie:**
   - GitHub Discussions
   - Reddit (r/MachineLearning)
   - Twitter/LinkedIn

---

**Última atualização:** 2025-09-30
**DOI:** https://zenodo.org/records/17171112
**License:** GNU GPLv3