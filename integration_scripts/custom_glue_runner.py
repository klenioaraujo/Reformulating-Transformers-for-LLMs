#!/usr/bin/env python3

import argparse
from glue import main as glue_main
from reformulated_transformers import ReformulatedTransformerForSequenceClassification

def run_glue_with_reformulated():
    parser = argparse.ArgumentParser()

    # Parâmetros originais do GLUE
    parser.add_argument("--data_dir", default=None, type=str, required=True)
    parser.add_argument("--model_type", default="bert", type=str)
    parser.add_argument("--model_name_or_path", default="bert-base-uncased", type=str)
    parser.add_argument("--task_name", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True)

    # Parâmetros específicos do modelo reformulado
    parser.add_argument("--use_reformulated", action="store_true")
    parser.add_argument("--reformulated_config", type=str, default=None)

    args = parser.parse_args()

    if args.use_reformulated:
        # Usar modelo reformulado
        from reformulated_transformers import ReformulatedConfig

        config = ReformulatedConfig.from_json_file(args.reformulated_config) if args.reformulated_config else ReformulatedConfig()
        model = ReformulatedTransformerForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            config=config,
            num_labels=2 if args.task_name == "cola" else 3 if args.task_name == "mnli" else 2
        )

        # Executar treinamento/evaluação
        run_training_with_custom_model(model, args)
    else:
        # Executar GLUE padrão
        glue_main.main(args)

def run_training_with_custom_model(model, args):
    """Adaptação do pipeline de treinamento para modelos reformulados"""

    # Implementar lógica de treinamento compatível
    # Isso pode requerer adaptar partes do código do GLUE-baselines

    from glue.utils import load_and_cache_examples, compute_metrics
    from transformers import Trainer, TrainingArguments

    # Carregar dados
    train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
    eval_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=True)

    # Configurar treinamento
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        num_train_epochs=3.0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )

    # Treinar e avaliar
    trainer.train()
    trainer.evaluate()

if __name__ == "__main__":
    run_glue_with_reformulated()