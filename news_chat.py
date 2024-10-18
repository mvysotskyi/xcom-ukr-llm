import asyncio
from pyppeteer import launch
from bs4 import BeautifulSoup

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer


async def extract_article_content_from_url(url):
    try:
        browser = await launch(
                handleSIGINT=False,
                handleSIGTERM=False,
                handleSIGHUP=False
        )
        page = await browser.newPage()


        await page.goto(url, {'waitUntil': 'networkidle0'})
        content = await page.content()

        soup = BeautifulSoup(content, 'html.parser')

        title_element = soup.find(attrs={"data-testid": "headline"})
        title = title_element.get_text(strip=True) if title_element else "Title not found"

        article_body = soup.find('section', {'name': 'articleBody'})

        if article_body:
            body_text = article_body.get_text(separator='\n', strip=True)
        else:
            body_text = "Article body not found"

        await browser.close()

        return {
            "title": title,
            "body": body_text
        }

    except Exception as e:
        return {"error": f"An error occurred: {e}"}



if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    model = AutoModelForCausalLM.from_pretrained("ukr_llm")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")

    messages_template = lambda title, body: [
        {"role": "user", "content": f"{body}"},
        {"role": "system", "content": "Imagine you are proukrainian activist. Be aggressive and biased, but do not start your messages with 'As a proukrainian activist ...'. Also use content above."},
        {"role": "user", "content": f"Title of news: {title}"}
    ]

    async def generate_response(url, _):
        article = await extract_article_content_from_url(url)

        messages = messages_template(article["title"], article["body"])

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = tokenizer([text], return_tensors="pt").to("cuda")

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response
    
    gr.ChatInterface(generate_response, concurrency_limit=20).launch()