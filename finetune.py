import torch

from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from transformers import TrainingArguments

from peft import LoraConfig
from peft import prepare_model_for_kbit_training, get_peft_model

from trl import SFTTrainer


class UkrLLM:
    def __init__(self, model_id="mistralai/Mistral-7B-Instruct-v0.3"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=self.bnb_config)
        self.model.gradient_checkpointing_enable()
        self.model = prepare_model_for_kbit_training(self.model)

    def __setup_lora(self):
        """
        Setup the LoRA optimizer
        """
        lora_config = LoraConfig(
            r=16,
            lora_alpha=16,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "lm_head",
            ],
            bias="none",
            lora_dropout=0.05,
            task_type="CAUSAL_LM",
        )

        self.model = get_peft_model(self.model, lora_config)


    def lora_train(self, dataset_path):
        """
        Train the model on the provided examples
        """
        self.__setup_lora()
        dataset = self.format_dataset(dataset_path)

        print("DEBUG: size of dataset - ", len(dataset))

        args = TrainingArguments(
            output_dir="lora_output",
            num_train_epochs=1,
            per_device_train_batch_size=16,
            learning_rate=1e-5,
            optim="sgd"
        )

        self.trainer = SFTTrainer(
            model=self.model,
            args=args,
            train_dataset=dataset,
            dataset_text_field='text',
            max_seq_length=512,
        )

        self.trainer.train()

    def fuse_lora(self):
        adapter_model = self.trainer.model

        merged_model = adapter_model.merge_and_unload()
        trained_tokenizer = self.trainer.tokenizer

        self.model = merged_model
        self.tokenizer = trained_tokenizer

    @staticmethod
    def format_prompts(examples, tokenizer: AutoTokenizer):
        """
        Define the format for your dataset
        This function should return a dictionary with a 'text' key containing the formatted prompts
        """
        messages = [
            {"role": "system", "content": "Imagine you are proukrainian activist. Be aggressive and biased."},
            {"role": "user", "content": f"Title of news: {examples['title']}"},
            {"role": "assistant", "content": f"{examples['answer']}"}
        ]

        return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}

    def format_dataset(self, csv_path):
        """
        Load the dataset from the provided CSV file
        """
        dataset = load_dataset('csv', data_files=csv_path, split='train')
        dataset = dataset.map(lambda x: self.format_prompts(x, self.tokenizer))

        dataset['text'][2]

        return dataset

if __name__ == "__main__":
    ukrllm = UkrLLM()
    ukrllm.lora_train("data/comments.csv")
    ukrllm.fuse_lora()

    ukrllm.model.save_pretrained("ukr_llm")