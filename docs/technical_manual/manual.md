# Manual Técnico de Construção - Sistema ΨQRH
## Reformulating Transformers for LLMs

### Fundação do Manual Técnico ΨQRH

Este é o manual técnico de construção do sistema ΨQRH (Reformulating Transformers), um projeto focado na reformulação de arquiteturas Transformer para modelos de linguagem de grande escala (LLMs).

#### Estrutura do Sistema de Construção

O sistema de construção é baseado em prompts JSON estruturados que definem ações atômicas para documentação, teste, integração, validação e atualização do projeto.

##### Regras Fundamentais

1. **Formato de Prompts**: Todo prompt deve estar em formato JSON válido, localizado em `construction_technical_manual/prompts/`
2. **Ações Atômicas**: Cada prompt representa uma ação específica: documentar, testar, integrar, validar ou atualizar
3. **Limpeza Automática**: Após execução, se `auto_delete: true`, o arquivo do prompt deve ser removido
4. **Estrutura de Diretórios**: Toda ação deve atualizar o arquivo `estrutura_diretorios.txt` com a estrutura atual usando `tree -d -L 3`
5. **Formato de Documentação**: O manual é escrito em Markdown com seções claras, referências a arquivos e explicações técnicas precisas
6. **Controle de Estado**: O estado de execução é mantido em `construction_technical_manual/state.json`

#### Arquivos de Controle

- **manual.md**: Este arquivo - documentação técnica acumulada
- **state.json**: Estado atual da construção e execução de prompts
- **estrutura_diretorios.txt**: Estrutura atual de diretórios do projeto
- **prompts/**: Diretório contendo prompts JSON para execução

#### Inicialização

Data de inicialização: 2024-05-20T12:00:00Z
Prompt fundacional: `000_foundation`
Status: Inicializado com sucesso

---

*Este manual é atualizado automaticamente durante o processo de construção do sistema ΨQRH.*