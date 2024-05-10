import gradio as gr


def convert_to_uppercase(text):
    return text.upper()


iface = gr.Interface(
    fn=convert_to_uppercase,
    inputs="text",
    outputs="text",
    title="文本转大写工具",
    description="输入任何文本，这个工具会将其转换为大写。"
)

iface.launch()
