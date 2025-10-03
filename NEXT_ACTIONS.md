# 🚀 Próximas Ações - ΨQRH FAIR Publication

**Status:** ✅ 100% Pronto para Publicação
**Data:** 2025-09-30
**Score FAIR:** 95/100

---

## ⚡ Ações Imediatas (Faça Hoje)

### 1. Corrigir Permissões
```bash
sudo ./scripts/fix_permissions.sh
```
**Por quê:** O diretório `models/` pertence ao root e precisa ser seu.

---

### 2. Criar Conta TestPyPI (5 minutos)
1. Acesse: https://test.pypi.org/account/register/
2. Preencha: nome, email, senha
3. Confirme email
4. **Importante:** Ative 2FA (obrigatório)

---

### 3. Gerar Token TestPyPI (2 minutos)
1. Acesse: https://test.pypi.org/manage/account/token/
2. Nome do token: `psiqrh-upload`
3. Escopo: `Entire account`
4. Copie o token (começa com `pypi-`)

---

### 4. Configurar Credenciais (3 minutos)
```bash
cat > ~/.pypirc << 'EOF'
[testpypi]
  username = __token__
  password = pypi-COLE_SEU_TOKEN_AQUI

[pypi]
  username = __token__
  password = pypi-OUTRO_TOKEN_DEPOIS
EOF

chmod 600 ~/.pypirc
```

---

### 5. Testar Upload (5 minutos)
```bash
source .venv/bin/activate
./scripts/publish_to_pypi.sh test
```

**Resultado esperado:**
```
✓ Model uploaded to https://test.pypi.org/project/psiqrh/
```

---

### 6. Testar Instalação (3 minutos)
```bash
# Em outro terminal/ambiente
python3 -m venv test_env
source test_env/bin/activate
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ psiqrh

python -c "import psiqrh; print(psiqrh.__version__)"
# Esperado: 1.0.0
```

---

## 📅 Ações Esta Semana

### 7. Criar Conta PyPI Real
- URL: https://pypi.org/account/register/
- Ative 2FA
- Gere token
- Adicione ao ~/.pypirc seção `[pypi]`

### 8. Publicar no PyPI
```bash
./scripts/publish_to_pypi.sh
```

### 9. Criar GitHub Release
```bash
git add .
git commit -m "FAIR compliance v1.0.0 - Ready for publication"
git tag -a v1.0.0 -m "FAIR-compliant release 1.0.0"
git push origin master
git push origin v1.0.0
```

### 10. Registrar no FAIRsharing
- Siga: `docs/FAIRSHARING_REGISTRATION.md`
- URL: https://fairsharing.org/accounts/signup

---

## 🎯 Ações Este Mês

### 11. Upload Modelo para HuggingFace
```python
from src.utils.model_hub import push_to_hub

# Supondo que você tenha um modelo treinado
push_to_hub(
    model=your_trained_model,
    repo_id="klenioaraujo/psiqrh-base",
    config=config
)
```

### 12. Submeter Paper Acadêmico
- Use template: `paper/PAPER_TEMPLATE.md`
- Adicione resultados experimentais
- Submeta para conferência/journal

### 13. Atualizar README com Badges
```markdown
[![PyPI](https://img.shields.io/pypi/v/psiqrh)](https://pypi.org/project/psiqrh/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17171112.svg)](https://doi.org/10.5281/zenodo.17171112)
[![License](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
```

---

## 📋 Checklist Completo

### Preparação (Completo ✅)
- [x] Metadata FAIR criado
- [x] Schemas JSON validados
- [x] Guias de reutilização testados
- [x] Build PyPI validado
- [x] Email atualizado
- [x] .gitignore configurado
- [x] Documentação completa

### Publicação (Pendente ⏳)
- [ ] Permissões corrigidas
- [ ] Conta TestPyPI criada
- [ ] Token TestPyPI gerado
- [ ] Credenciais configuradas
- [ ] Upload TestPyPI realizado
- [ ] Instalação TestPyPI testada
- [ ] Conta PyPI criada
- [ ] Upload PyPI realizado
- [ ] GitHub release criado
- [ ] FAIRsharing registrado

### Divulgação (Futuro 🔮)
- [ ] HuggingFace model uploaded
- [ ] Paper submitted
- [ ] README updated with badges
- [ ] Blog post written
- [ ] Social media announcement

---

## 🆘 Links Rápidos

**Documentação:**
- 📖 Guia Completo: `MANUAL_STEPS.md`
- ⚡ Quick Start: `QUICK_START_FAIR.md`
- 📊 FAIR Compliance: `FAIR_COMPLIANCE.md`

**Registro:**
- TestPyPI: https://test.pypi.org/account/register/
- PyPI: https://pypi.org/account/register/
- FAIRsharing: https://fairsharing.org/accounts/signup
- HuggingFace: https://huggingface.co/join
- ORCID: https://orcid.org/register

**Scripts:**
- `./scripts/fix_permissions.sh` - Corrigir permissões
- `./scripts/publish_to_pypi.sh` - Publicar pacote
- `./scripts/update_orcid.py` - Atualizar ORCID
- `./scripts/validate_schemas.py` - Validar schemas
- `./scripts/test_reuse_guides.py` - Testar guias

---

## 💡 Dicas

### Para TestPyPI
- Use sempre `--extra-index-url https://pypi.org/simple/` ao instalar
- TestPyPI é resetado periodicamente (não é permanente)
- Use para testar antes de publicar no PyPI real

### Para PyPI Real
- **Não pode deletar** pacotes publicados
- **Não pode reutilizar** versões (1.0.0 é permanente)
- Publique apenas quando tiver certeza
- Teste tudo no TestPyPI primeiro

### Para Tokens
- Guarde em local seguro (gerenciador de senhas)
- Nunca commite no git
- Use tokens específicos por projeto (não "entire account")
- Regenere se suspeitar de comprometimento

---

## ✅ Como Saber que Deu Certo

### TestPyPI
```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ psiqrh
python -c "import psiqrh; print('✅ Funcionou!')"
```

### PyPI Real
```bash
pip install psiqrh
python -c "import psiqrh; print('✅ Publicado com sucesso!')"
```

### GitHub Release
- Acesse: https://github.com/klenioaraujo/Reformulating-Transformers-for-LLMs/releases
- Deve ver: `v1.0.0` listado

### FAIRsharing
- Deve receber email de confirmação
- Registro aparecerá em https://fairsharing.org/search

---

## 🎉 Quando Tudo Estiver Completo

1. **Comemore!** 🍾 Você criou um projeto FAIR-compliant!

2. **Compartilhe:**
   - GitHub Discussions
   - Reddit r/MachineLearning
   - Twitter/LinkedIn
   - Comunidade científica

3. **Mantenha:**
   - Responda issues
   - Aceite pull requests
   - Atualize documentação
   - Publique novas versões

---

**Lembre-se:** Faça um passo de cada vez. Comece com TestPyPI!

**Última atualização:** 2025-09-30
**DOI:** https://zenodo.org/records/17171112
**Email:** klenioaraujo@gmail.com