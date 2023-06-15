package ai.mlc.mlcchat

import android.os.Looper

class ChatModule {
    private var text = ""
    private var count = 0

    init {
    }

    fun unload() {
        require(!Looper.getMainLooper().isCurrentThread)
        text = ""
        count = 0
        Thread.sleep(1000)
    }

    fun reload(modelLib: String, modelPath: String) {
        require(!Looper.getMainLooper().isCurrentThread)
        text = ""
        count = 0
        Thread.sleep(1000)
    }

    fun resetChat() {
        require(!Looper.getMainLooper().isCurrentThread)
        text = ""
        count = 0
        Thread.sleep(1000)
    }

    fun prefill(input: String) {
        require(!Looper.getMainLooper().isCurrentThread)
        text = input
        count = 0
        Thread.sleep(1000)
    }

    fun getMessage(): String {
        require(!Looper.getMainLooper().isCurrentThread)
        return text.repeat(count)
    }

    fun runtimeStatsText(): String {
        require(!Looper.getMainLooper().isCurrentThread)
        return text
    }

    fun evaluate() {

    }

    fun stopped(): Boolean {
        require(!Looper.getMainLooper().isCurrentThread)
        return count >= 10
    }

    fun decode() {
        require(!Looper.getMainLooper().isCurrentThread)
        ++count
        Thread.sleep(1000)
    }

}