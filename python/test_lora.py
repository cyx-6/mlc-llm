from mlc_chat import ChatModule
from mlc_chat.callback import StreamToStdout

cm = ChatModule(model="llama-2-7b-chat-hf-q0f16")
# cm.generate(
#     prompt="北京在哪里",
#     progress_callback=StreamToStdout(callback_interval=2),
# )
# print(f"Statistics: {cm.stats()}\n")
# cm.reset_chat()

cm.apply_lora("/home/ycai/models/Llama2-Chinese-7b-Chat-LoRA")
cm.generate(
    prompt="北京在哪里",
    progress_callback=StreamToStdout(callback_interval=2),
)
print(f"Statistics: {cm.stats()}\n")
