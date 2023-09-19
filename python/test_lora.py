from mlc_chat import ChatModule
from mlc_chat.callback import StreamToStdout

cm = ChatModule(model="llama-2-7b-chat-hf-q4f16_ft")
print("===== without lora =====")
cm.generate(
    prompt="write a long description of the united states please",
    progress_callback=StreamToStdout(callback_interval=2),
)
print(f"Statistics: {cm.stats()}\n")
cm.reset_chat()
