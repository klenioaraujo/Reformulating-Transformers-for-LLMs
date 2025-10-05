# ΨQRH Security Directory Structure

## 🗂️ Estrutura de Diretórios Implementada

### Diretório de Ativos Seguros (Isolado)
```
data/secure_assets/
├── Ψcws/                    # Arquivos .Ψcws criptografados e certificados
│   ├── test-personal.Ψcws
│   └── test-enterprise.Ψcws
├── manifests/               # Manifestos de auditoria
│   ├── test-personal.manifest.json
│   └── test-enterprise.manifest.json
├── certificates/            # Certificações digitais
│   ├── test-personal.certificate.json
│   └── test-enterprise.certificate.json
└── audit_log.jsonl          # Log de auditoria
```

### Diretório de Ativos Existentes (Não Seguros)
```
data/
├── Ψcws/                    # Arquivos .Ψcws existentes (não certificados)
│   ├── integration_test.Ψcws
│   ├── philosophy.Ψcws
│   └── d41d8cd98f00b204e9800998ecf8427e.Ψcws
└── secure_assets/           # Diretório isolado para ativos seguros
```

## 🔒 Separação de Responsabilidades

### Ativos Seguros (`data/secure_assets/`)
- **Certificação Obrigatória**: Todos os arquivos são certificados
- **Validação de Segurança**: Requer validação antes do uso
- **Auditoria**: Logs detalhados para níveis enterprise/government
- **Controle de Acesso**: Chaves obrigatórias para acesso

### Ativos Existentes (`data/Ψcws/`)
- **Sem Certificação**: Arquivos originais do sistema
- **Acesso Direto**: Sem validação de segurança
- **Compatibilidade**: Mantém compatibilidade com sistema existente

## 🚀 Comandos Disponíveis

### Para Ativos Seguros
```bash
# Criar ativo seguro
make new-secure-asset SOURCE=file.txt NAME=asset LEVEL=enterprise KEY=secret

# Listar ativos seguros
make list-secure-assets

# Validar ativo seguro
make validate-secure-asset NAME=asset KEY=secret

# Treinar com ativo seguro
make train-with-secure-asset NAME=asset KEY=secret
```

### Para Ativos Existentes
```bash
# Listar todos os arquivos .Ψcws (incluindo não certificados)
make list-Ψcws
```

## 🛡️ Benefícios da Separação

1. **Segurança**: Ativos sensíveis isolados em diretório protegido
2. **Compatibilidade**: Sistema existente continua funcionando
3. **Clareza**: Separação clara entre dados seguros e não seguros
4. **Auditoria**: Controle total sobre ativos certificados
5. **Escalabilidade**: Estrutura pronta para expansão

## 📊 Status Atual

- ✅ **Ativos Seguros**: 2 arquivos certificados em `data/secure_assets/`
- ✅ **Ativos Existentes**: 11 arquivos em `data/Ψcws/`
- ✅ **Sistema Funcional**: Todos os comandos operacionais
- ✅ **Separação Completa**: Diretórios isolados funcionando

A estrutura agora garante que apenas arquivos `.Ψcws` certificados podem ser usados para treinamento seguro, enquanto mantém compatibilidade total com o sistema existente.