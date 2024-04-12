from openai import OpenAI
import requests


client = OpenAI(api_key='Replace with your own OPENAI KEY.')

def dalle_gen(client, saved_path, input_text, saved=False):
    try:

        if len(input_text) > 1000:
            input_text = input_text[:1000]

        response = client.images.generate(
        model="dall-e-2",
        prompt=input_text,
        size="1024x1024",
        quality="standard",
        n=1,
        )

        image_url = response.data[0].url
        response = requests.get(image_url)

        if response.status_code == 200:
            if saved:
                with open(saved_path, 'wb') as f:
                    f.write(response.content)

            print(f"Saved to {saved_path}")
            return saved_path
        else:
            print("Fail...")

    except Exception as e:

        print(f"An error occurred: {e}")

        return None


def get_cls_index_name(label_index):
    with open("data_txt/ImageNet_LT/ImageNet_cls_name.txt", "r") as file:
        labels = [label.strip('",') for label in file.read().splitlines()]
    
    if 0 <= label_index < len(labels):
        return labels[label_index]
    else:
        return "Index out of range"



def description_refine(input_text, cls_name):
    user_content = "This description does not seem to be representative of the class " + cls_name + ". Could you refine it to enhance the distinctive features of class " + cls_name

    completion = client.chat.completions.create(
    # model="gpt-3.5-turbo",
    model="gpt-4-turbo-preview",
    messages=[
        {"role": "user", "content": input_text},
        {"role": "user", "content": user_content}
    ]
    )

    output = completion.choices[0].message.content

    return output


def get_cls_template(cls_name, cls_index, filename="data_txt/ImageNet_LT/class_templates.txt"):
    try:
        with open(filename, "r") as file:
            for line in file:
                index, saved_template = line.strip().split(':', 1)
                if int(index) == cls_index:
                    return saved_template 
    except FileNotFoundError:
        pass 

    template = "Template: A photo of the class " + cls_name + " with {feature 1}{feature 2}{...}."
    user_content = "Please use the Template to summarize the most distinctive features of class " + cls_name

    completion = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "user", "content": template},
            {"role": "user", "content": user_content}
        ]
    )

    output = completion.choices[0].message.content

    # 保存新生成的模板到文件
    with open(filename, "a") as file:  # 使用追加模式'a'
        file.write(f"{cls_index}:{output}\n")

    return output

