from mlc_chat import ChatModule
from mlc_chat.callback import StreamToStdout

cm = ChatModule(model="llama-2-7b-chat-hf-q4f16_1")
print("===== without lora =====")
cm.generate(
    prompt="what is the united states",
    progress_callback=StreamToStdout(callback_interval=2),
)
print(f"Statistics: {cm.stats()}\n")
cm.reset_chat()
