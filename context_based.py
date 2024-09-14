import gradio as gr

from comments_reader import CommentsReader
from transformers import AutoModelForCausalLM, AutoTokenizer


if __name__ == "__main__":
    device = "cuda"

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-1.5B-Instruct",
        torch_dtype="auto",
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")
    cr = CommentsReader("data/ukraine_comments.json")
    text = cr.nrand_formatted(20)

    messages = [
        {"role": "system", "content": "You are sevierly pro-ukrainian reddit user. I give you a examples of topics and pro-ukrainian comments from reddit. "},
        {"role": "system", "content": "TOPICS and COMMENTS:\n" + text},
        {"role": "system", "content": "User will give you topic. Make very pro-ukrainian and detailed comment to this topic. Use the comments above as a source of inspiration and be very creative."},
        {"role": "user", "content": None}
    ]

    def generate_response(topic, _):
        messages[-1]["content"] = topic

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response 

    gr.ChatInterface(generate_response).launch()