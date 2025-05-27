import gradio as gr

from comments_reader import CommentsReader
from transformers import AutoModelForCausalLM, AutoTokenizer


if __name__ == "__main__":
    device = "cuda"

    model = AutoModelForCausalLM.from_pretrained(
        "ukr_llm"
    )

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
    cr = CommentsReader("data/ukraine_comments.json")
    text = cr.nrand_formatted(20)

    messages = [
        {"role": "system", "content": "Imagine you are proukrainian activist. Be aggressive and biased, but do not start your messages with 'As a proukrainian activist ...'. Also use content above. Make dark jokes about Vladimir Putin all the time."},
        {"role": "user", "content": None},
    ]

    def generate_response(topic, _):
        global text
        # original = "\nUser will give you topic. Write me a short and very pro-ukrainian with dark jokes about Vladimir Putin and detailed comment to this topic.\n"
        messages[-1]["content"] = f"Title of news: {topic}"

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=1024
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response


    gr.ChatInterface(generate_response).launch()