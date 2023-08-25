from mlc_chat import ChatModule
from mlc_chat.callback import StreamToStdout

cm = ChatModule(model="llama-2-7b-chat-hf-q4f16_1")
print("===== without lora =====")
output = cm.generate(
    prompt="where is the united states",
    progress_callback=StreamToStdout(callback_interval=2),
)
cm.reset_chat()

print("===== with lora =====")
cm.apply_lora("/home/ycai/test/model/Llama2-Chinese-7b-Chat-LoRA")
output = cm.generate(
    prompt="where is the united states",
    progress_callback=StreamToStdout(callback_interval=2),
)
